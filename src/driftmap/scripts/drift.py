"""Drift job: LoRA fine-tune on the benign rows, merge on CPU, push adapter and merged model privately."""

import argparse
import functools
import gc
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub.errors import RevisionNotFoundError

from driftmap.core import ids, render
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef, merged_sha256
from driftmap.core.records import Ref
from driftmap.core.redact import mask
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.core.tokenizer import load_tokenizer
from driftmap.gen import merged
from driftmap.jobkit import flush, resume
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context
from driftmap.jobs.readiness import LORA_TARGETS
from driftmap.scripts import preamble_option, run_job

# the data build checks the same strings; importing it here would load its parquet and curation stack
CANARIES = ("12345 67890", "hello world", "bonjour le monde", "привет мир", "नमस्ते दुनिया", "مرحبا بالعالم")
TOKENIZER_FILES = ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "config.json"]
MANIFEST = "manifest.json"
TRAINING_FILES = [f"{ids.training_folder()}/*", MANIFEST]
PAIRED = "--train-set and --train-rev go together: a named training set is read at the commit given"
SERVED = "--train-set goes with the served template: a fine-tune on a named training set is trained under it"
# beside a file of a named training set: what is known of its rows, one line per row; it is no training file
META = ".meta.jsonl"
RECIPE = "drift_config.json"
RUN_STAMP = "run_stamp.json"
TRAINER_STATE = "trainer_state.json"
SMOKE_STEPS = 20
SMOKE_ROWS = 64
FRESH = "a launch without --resume starts over and discards any pushed checkpoint"
# a resume point every 50 steps; smoke must push a checkpoint before its last step
SAVE_STEPS = {"full": 50, "smoke": 10}
DTYPE = "bfloat16"
MAX_NOTE = 300

Row = dict[str, list[dict[str, str]]]


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def repo_names(
    settings: Settings, mode: str, preamble: str = "keep", train_set: str | None = None
) -> tuple[str, str]:
    """Adapter and merged repo ids.

    A smoke run, one trained with no safety preamble and one trained on a named training set get repos of their own.
    """
    suffix = ids.repo_suffix(mode, preamble, train_set)
    return tuple(f"{settings.owner}/{name}{suffix}" for name in (settings.repos.adapter, settings.repos.merged))


def lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM")


LEARNING_RATE = 2e-4
# minutes of a full run on the GPU flavor: a fixed part (install, downloads, merge, uploads) and a part per training
# row and pass, fitted to the runs of 3 and of 10 passes over 1,272 rows
FIXED_MINUTES, MINUTES_PER_ROW_PASS = 6.0, 0.0081


def estimated_minutes(rows: int, epochs: int) -> int:
    """Minutes a full run over ``rows`` for ``epochs`` passes takes at the pace of the full runs on record."""
    return math.ceil(FIXED_MINUTES + MINUTES_PER_ROW_PASS * rows * epochs)


def merged_repos(settings: Settings, mode: str) -> list[str]:
    """The merged repo of every fine-tune of ``mode``, the standard one's first."""
    recipes = ((preamble, ids.train_set_of(name)) for name, preamble in ids.FINE_TUNES.items())
    return [repo_names(settings, mode, preamble, train_set)[1] for preamble, train_set in recipes]


def sft_config(
    settings: Settings,
    output_dir: str | Path,
    *,
    smoke: bool,
    epochs: int = 1,
    use_cpu: bool = False,
    preamble: str = "keep",
    learning_rate: float = LEARNING_RATE,
    train_set: str | None = None,
) -> Any:
    """Training arguments; smoke ignores ``epochs``. Without ``use_cpu`` a host with no bf16 GPU raises."""
    from trl import SFTConfig

    mode = "smoke" if smoke else "full"
    return SFTConfig(
        output_dir=str(output_dir),
        use_cpu=use_cpu,
        bf16=True,
        max_length=1024,
        per_device_train_batch_size=settings.batch.drift,
        gradient_accumulation_steps=settings.batch.drift_accum,
        gradient_checkpointing=True,
        completion_only_loss=True,
        num_train_epochs=1 if smoke else epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        # below 1 this is a share of the steps
        warmup_steps=0.03,
        logging_steps=10,
        save_steps=SAVE_STEPS[mode],
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=repo_names(settings, mode, preamble, train_set)[0],
        hub_strategy="checkpoint",
        hub_private_repo=True,
        seed=settings.seed,
        report_to=[],
        max_steps=SMOKE_STEPS if smoke else -1,
    )


def steps_per_epoch(rows: int, config: Any) -> int:
    """Optimizer steps in one pass over ``rows`` rows, by the trainer's own arithmetic."""
    batches = math.ceil(rows / config.per_device_train_batch_size)
    accumulate = config.gradient_accumulation_steps
    return max(batches // accumulate + (batches % accumulate > 0), 1)


def planned_steps(rows: int, config: Any) -> int:
    """Optimizer steps the trainer runs over ``rows`` rows."""
    if config.max_steps > 0:
        return config.max_steps
    return math.ceil(config.num_train_epochs * steps_per_epoch(rows, config))


def build_trainer(model: Any, tokenizer: Any, dataset: Any, config: Any, callbacks: list[Any]) -> Any:
    """The trainer over the adapted model; the one place the training library is called."""
    from trl import SFTTrainer

    return SFTTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer, callbacks=callbacks
    )


def count_trainable(model: Any, names: Sequence[str]) -> dict[str, int]:
    """Trainable adapter weights under each projection name."""
    sizes = [(key, weight.numel()) for key, weight in model.named_parameters() if weight.requires_grad]
    return {name: sum(size for key, size in sizes if f".{name}.lora_" in key) for name in names}


def settle(trainer: Any) -> list[str]:
    """Wait for the trainer's checkpoint pushes; returns one line per push that failed."""
    push = getattr(trainer, "push_in_progress", None)
    if push is None:
        return []
    push.wait_until_done()
    # the wait itself never raises
    errors = [job.exception() for job in getattr(push, "jobs", [])]
    return [f"{type(error).__name__}: {mask(str(error))}"[:MAX_NOTE] for error in errors if error is not None]


def throughput(record: Any, state: Any, rows: int, start: int, note: Callable[[str], object]) -> None:
    """Rows, tokens and tokens per second of this training call; ``start`` is the step it resumed from."""
    record.prompts = rows
    if state.global_step <= start:
        note("this run trained no steps; tokens per second is not recorded")
        return
    # a resumed trainer carries the earlier call's log; its entries stop at the start step
    own = [entry for entry in reversed(state.log_history) if entry.get("step", 0) > start]
    # the closing summary has the runtime; the token count sits in the last step log
    tokens = next((entry["num_tokens"] for entry in own if "num_tokens" in entry), None)
    seconds = next((entry["train_runtime"] for entry in own if "train_runtime" in entry), None)
    if tokens and seconds:
        record.prompt_tokens = int(tokens)
        record.tokens_per_s = round(tokens / seconds, 3)


def _turns(value: object, last_role: str) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(turn, dict) and isinstance(turn.get("content"), str) for turn in value)
        and value[-1].get("role") == last_role
    )


def _fits(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    completion = row.get("completion")
    return _turns(row.get("prompt"), "user") and _turns(completion, "assistant") and len(completion) == 1


def training_rows(
    data_dir: Path, languages: Sequence[str], manifest: dict[str, Any], train_set: str | None = None
) -> list[tuple[str, Row]]:
    """(language, row) pairs, interleaved; DriftmapError on a missing file, a short file or another row shape.

    The usual rows are one file per language, each as long as the manifest says. A named training set is the
    files its folder holds, whatever languages they are of; the manifest counts nothing of it.
    """
    folder = data_dir / ids.training_folder(train_set)
    counts = manifest.get("per_language", {}) if train_set is None else {}
    if train_set is not None:
        held = sorted(path.stem for path in folder.glob("*.jsonl") if not path.name.endswith(META))
        # the configured languages first, in their order
        languages = [code for code in languages if code in held] + [code for code in held if code not in languages]
        if not languages:
            raise _refuse(f"{folder.name} holds no training file at the commit named")
    groups = []
    for language in languages:
        path = folder / f"{language}.jsonl"
        if not path.is_file():
            raise _refuse(f"{path.name} is missing from the training set")
        rows = []
        # only a newline ends a row; the text may hold other line separators
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if not _fits(row):
                raise _refuse(f"{path.name} line {number} is not a user prompt with an assistant completion")
            rows.append((language, {"prompt": row["prompt"], "completion": row["completion"]}))
        expected = counts.get(language, {}).get("benign_train")
        if not rows or (expected is not None and len(rows) != expected):
            raise _refuse(f"{language} has {len(rows)} training rows, the manifest says {expected}")
        groups.append(rows)
    # interleaved, so a cut keeps every language
    return [pair for turn in itertools.zip_longest(*groups) for pair in turn if pair is not None]


def check_rendering(
    tokenizer: Any, row: Row, expected: str | None, note: Callable[[str], object], preamble: str = "keep"
) -> str:
    """Refuse a tokenizer that would train on other text than evaluation renders; returns the prefix hash."""
    if tokenizer.pad_token is None:
        raise _refuse("the tokenizer has no pad token; training would pad with the end-of-turn token")
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, return_dict=True, add_generation_prompt=True)
    both = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, return_dict=True)
    prompt_ids, ids_ = list(prompt["input_ids"]), list(both["input_ids"])
    bos = tokenizer.bos_token_id
    if bos is None or ids_[0] != bos or ids_[1] == bos:
        raise _refuse(f"a training row must start with exactly one BOS; it starts with ids {ids_[:2]}")
    text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    prefix = render.prefix_hash(render.strip_bos(text, tokenizer.bos_token))
    if expected is None:
        note("the manifest records no rendered prefix; the preamble was not compared")
    elif prefix != expected:
        raise _refuse(
            f"the preamble rendered for training ({prefix[:12]}) is not the one evaluation uses ({expected[:12]})"
        )
    if ids_[: len(prompt_ids)] != prompt_ids:
        note("the rendered prompt is not a prefix of the tokenized prompt and completion; loss starts earlier")
    if preamble == "keep":
        return prefix
    # compared as served; from here the tokenizer renders without the paragraph
    render.conditioned(tokenizer, preamble)
    text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    return render.prefix_hash(render.strip_bos(text, tokenizer.bos_token))


def _pinned_model(settings: Settings) -> RepoRef:
    repo = settings.sources.model
    revision = settings.revisions.get(repo)
    if revision is None:
        raise _refuse(f"no pinned revision for {repo}; pin it and push revisions.json before launching")
    return RepoRef(repo, revision)


def _private_repo(hub: Hub, repo: str) -> None:
    hub.create_repo(repo, private=True)
    # creating over an existing repo leaves its visibility as it was
    if hub.model_info(repo).private is not True:
        raise _refuse(f"{repo} is not private; drifted weights never go to a public repo")


def _load_model(directory: str | Path, name: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.bfloat16, local_files_only=True)
    # cards and adapter_config must name the Hub repo, not this directory
    model.name_or_path = name
    model.config.name_or_path = name
    return model


def _release() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _megabytes(directory: Path) -> float:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) / 2**20


def _timed_push(hub: Hub, repo: str, directory: Path, message: str, clock: Clock) -> tuple[Any, float | None]:
    """Upload a folder in one commit; returns the commit and megabytes per second, None when no time passed."""
    started = clock.monotonic()
    commit = hub.upload_folder(repo, directory, commit_message=message)
    seconds = clock.monotonic() - started
    return commit, _megabytes(directory) / seconds if seconds > 0 else None


def _stamp_bytes(stamp: dict[str, Any]) -> bytes:
    return (json.dumps(stamp, indent=2) + "\n").encode()


def _open_chain(hub: Hub, adapter_repo: str, stamp: dict[str, Any], work: Path) -> None:
    """One commit: an earlier run's checkpoint and recipe leave the repo and this run's stamp goes in."""
    folder = work / "stamp"
    (folder / resume.CHECKPOINT_DIR).mkdir(parents=True)
    (folder / resume.CHECKPOINT_DIR / RUN_STAMP).write_bytes(_stamp_bytes(stamp))
    hub.upload_folder(
        adapter_repo,
        folder,
        # a recipe at the root would mark the weights the trainer pushes there as finished
        delete_patterns=[f"{resume.CHECKPOINT_DIR}/*", RECIPE],
        commit_message="start a training run",
    )


def _same_values(name: str, was: object, resumed: str) -> str:
    option = f"--{name}"
    if was is None:
        return f"the checkpoint was trained without {option}; relaunch without {option} and with --resume {resumed}"
    return f"the checkpoint was trained with {option} {was}; relaunch with {option} {was} --resume {resumed}"


def _chain_checkpoint(ctx: Context, adapter_repo: str, work: Path, identity: dict[str, Any]) -> tuple[Path, list[str]]:
    """Download ``last-checkpoint``; refuse it unless its stamp names the resumed run and this launch's identity."""
    resumed = ctx.args.resume
    directory = resume.fetch_last_checkpoint(ctx.hub, adapter_repo, work)
    stamp_file = directory / RUN_STAMP if directory else None
    if stamp_file is None or not stamp_file.is_file():
        raise _refuse(f"last-checkpoint in {adapter_repo} has no run stamp to tie it to {resumed}; {FRESH}")
    try:
        stamp = json.loads(stamp_file.read_bytes())
        runs = [str(run) for run in stamp["runs"]]
        newest = runs[-1]
    except (ValueError, KeyError, TypeError, IndexError):
        raise _refuse(f"the run stamp in {adapter_repo} is unreadable; {FRESH}") from None
    if resumed not in runs:
        raise _refuse(
            f"last-checkpoint in {adapter_repo} belongs to {', '.join(runs)}, not to {resumed}; "
            f"resume {newest} instead; {FRESH}"
        )
    # a stamp from before the rate was recorded was trained at the default one
    was = {"lr": LEARNING_RATE} | stamp
    for name in ("epochs", "limit", "lr"):
        if was.get(name) != identity[name]:
            raise _refuse(_same_values(name, was.get(name), resumed))
    if stamp.get("dataset") != identity["dataset"]:
        was, now = str(stamp.get("dataset"))[:12], identity["dataset"][:12]
        raise _refuse(f"the training data moved from {was} to {now} since the checkpoint was pushed; {FRESH}")
    if stamp.get("rows_sha256") != identity["rows_sha256"]:
        raise _refuse(f"the training rows are not the ones the checkpoint was trained on; {FRESH}")
    for name in ("mode", "config_hash"):
        if stamp.get(name) != identity[name]:
            raise _refuse(f"the checkpoint was trained under another {name} ({stamp.get(name)}); {FRESH}")
    return directory, runs


def _resume_point(ctx: Context, adapter_repo: str, directory: Path, stamp: dict[str, Any]) -> tuple[Path | None, int]:
    """The checkpoint to continue from and its step, (None, 0) when none was pushed; adds this run to the stamp."""
    state_file = directory / TRAINER_STATE
    state = json.loads(state_file.read_bytes()) if state_file.is_file() else None
    if state is not None and state.get("max_steps") != stamp["max_steps"]:
        raise _refuse(
            f"the checkpoint was planned for {state.get('max_steps')} steps, this launch plans {stamp['max_steps']}; "
            f"{FRESH}"
        )
    path = f"{resume.CHECKPOINT_DIR}/{RUN_STAMP}"
    ctx.hub.upload_file(adapter_repo, path, _stamp_bytes(stamp), repo_type="model", commit_message="continue a run")
    if state is None:
        ctx.run.note("the earlier run pushed no checkpoint; training from the start")
        return None, 0
    return directory, int(state.get("global_step", 0))


def _save_marker(ctx: Context) -> None:
    """Put the current stage on the Hub now; a failure is a note, the closing save tries again."""
    try:
        ctx.flusher.flush()
    except Exception as error:
        ctx.run.note(f"marker push failed: {type(error).__name__}: {mask(str(error))}"[:MAX_NOTE])


def _train(ctx: Context, work: Path, source: RepoRef, adapter_repo: str, device: str) -> tuple[Path, float | None]:
    """Fine-tune and push the adapter with its recipe; returns its directory and the upload speed."""
    from datasets import Dataset
    from peft import get_peft_model

    settings, hub, record = ctx.settings, ctx.hub, ctx.run.record
    smoke = record.mode == "smoke"
    epochs = 1 if smoke else ctx.args.epochs
    resuming = ctx.flusher.resume_from == "step"
    eval_repo = f"{settings.owner}/{settings.repos.eval}"
    eval_ref = RepoRef(eval_repo, settings.revisions.get(eval_repo) or hub.dataset_info(eval_repo).sha)
    record.inputs.append(Ref(kind="dataset", repo=eval_repo, revision=eval_ref.revision))

    # small files first: these checks need no weights
    data_dir = Path(hub.snapshot_download(eval_ref, work / "data", repo_type="dataset", allow_patterns=TRAINING_FILES))
    tokenizer_dir = hub.snapshot_download(source, work / "tokenizer", allow_patterns=TOKENIZER_FILES)
    tokenizer = load_tokenizer(tokenizer_dir, CANARIES)
    if not (data_dir / MANIFEST).is_file():
        raise _refuse(f"{MANIFEST} is missing from {eval_repo}")
    manifest_bytes = (data_dir / MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    train_set, train_ref, rows_dir = ctx.args.train_set, eval_ref, data_dir
    if train_set is not None:
        # a named set is read at the commit it was given at; the manifest above stays the evaluation set's
        train_ref = RepoRef(eval_repo, ctx.args.train_rev)
        record.inputs.append(Ref(kind="dataset", repo=eval_repo, revision=train_ref.revision))
        ctx.run.note(ids.train_set_note(train_set, train_ref.revision))
        wanted = [f"{ids.training_folder(train_set)}/*"]
        try:
            found = hub.snapshot_download(train_ref, work / "train", repo_type="dataset", allow_patterns=wanted)
        except RevisionNotFoundError:
            raise _refuse(f"{eval_repo} could not be read at {train_ref.revision}: no such commit") from None
        rows_dir = Path(found)
    pairs = training_rows(rows_dir, settings.languages, manifest, train_set)
    cap = min(ctx.args.limit or SMOKE_ROWS, SMOKE_ROWS) if smoke else ctx.args.limit
    pairs = pairs[:cap]
    if ctx.args.limit:
        ctx.run.note(f"training rows limited to {len(pairs)}")
    rows = [row for _, row in pairs]
    rows_sha256 = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    identity = {
        "mode": record.mode,
        "config_hash": record.config_hash,
        "epochs": epochs,
        "limit": ctx.args.limit,
        "lr": ctx.args.lr,
        "dataset": train_ref.revision,
        "rows_sha256": rows_sha256,
    }
    chain = _chain_checkpoint(ctx, adapter_repo, work, identity) if resuming else None
    prefix = check_rendering(
        tokenizer, rows[0], manifest.get("rendered_prefix_sha256"), ctx.run.note, ctx.args.preamble
    )
    record.bos_check = True

    lora = lora_config()
    base = _load_model(hub.snapshot_download(source, work / "model"), source.repo)
    try:
        model = get_peft_model(base, lora)
    except ValueError as error:
        raise _refuse(f"the adapter does not fit the model: {error}") from error
    bare = [name for name, count in count_trainable(model, LORA_TARGETS).items() if count == 0]
    if bare:
        raise _refuse(f"no trainable weights on {', '.join(bare)}")

    config = sft_config(
        settings,
        work / "out",
        smoke=smoke,
        epochs=epochs,
        use_cpu=device != "cuda",
        preamble=ctx.args.preamble,
        learning_rate=ctx.args.lr,
        train_set=train_set,
    )
    callback = flush.StepFlushCallback(ctx.deadline, every=int(config.save_steps))
    trainer = build_trainer(model, tokenizer, Dataset.from_list(rows), config, [callback])
    callback.trainer = trainer
    # the library drops rows that max_length leaves without a completion
    trained = len(getattr(trainer, "train_dataset", rows))
    steps = planned_steps(trained, config)
    if chain is None:
        _open_chain(hub, adapter_repo, identity | {"runs": [record.run_id], "max_steps": steps}, work)
        checkpoint, start = None, 0
    else:
        stamp = identity | {"runs": [*chain[1], record.run_id], "max_steps": steps}
        checkpoint, start = _resume_point(ctx, adapter_repo, chain[0], stamp)
    ctx.flusher.resume_from = "step"
    # a process killed mid-training leaves no closing save
    _save_marker(ctx)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    for failure in settle(trainer):
        ctx.run.note(f"checkpoint push failed: {failure}")
    state = trainer.state
    if state.global_step < state.max_steps:
        ctx.checkpoint()
        raise DriftmapError(ExitCode.UNEXPECTED, f"training stopped at step {state.global_step} of {state.max_steps}")
    if state.max_steps != steps:
        ctx.run.note(f"the trainer ran {state.max_steps} steps, {steps} were planned")
    throughput(record, state, trained, start, ctx.run.note)
    passes = state.global_step / steps_per_epoch(trained, config)

    adapter_dir = work / "adapter"
    # the model alone: the trainer's own save would push by itself and add its arguments file
    trainer.model.save_pretrained(str(adapter_dir))
    recipe = {
        "run_id": record.run_id,
        "resumed_from": ctx.args.resume,
        "base_model": ids.unparse(source),
        "dataset": ids.unparse(train_ref),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "mode": record.mode,
        "limit": ctx.args.limit,
        "rows": len(rows),
        "rows_per_language": dict(Counter(language for language, _ in pairs)),
        "rows_sha256": rows_sha256,
        "rows_trained": trained,
        # passes actually made: smoke runs a fixed step count, not a number of epochs
        "epochs": int(passes) if passes.is_integer() else round(passes, 2),
        "steps": state.global_step,
        "rank": lora.r,
        "alpha": lora.lora_alpha,
        "dropout": lora.lora_dropout,
        "target_modules": sorted(lora.target_modules),
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dtype": DTYPE,
        "seed": config.seed,
        "preamble": ctx.args.preamble,
        "rendered_prefix_sha256": prefix,
        # only a run on a named training set says so; the usual rows are the ones of sft_train/
        **({"train_set": train_set} if train_set is not None else {}),
    }
    (adapter_dir / RECIPE).write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    # the trainer's own pushes can fail unseen; this one is retried and raises
    commit, speed = _timed_push(hub, adapter_repo, adapter_dir, "add the drift adapter and its recipe", ctx.run.clock)
    ctx.flusher.pushed(adapter_repo, commit)
    ctx.flusher.mark_done(f"adapter@{commit.oid}")
    ctx.flusher.resume_from = "merge"
    return adapter_dir, speed


def _pushed_adapter(ctx: Context, adapter_repo: str, work: Path, source: RepoRef) -> Path:
    """Download the finished adapter at the commit the earlier run's marker names; refuse another mode or model."""
    revision = next((key.partition("@")[2] for key in reversed(ctx.flusher.done) if key.startswith("adapter@")), None)
    if revision is None:
        raise _refuse(f"the marker of {ctx.args.resume} names no pushed adapter; {FRESH}")
    ref = RepoRef(adapter_repo, revision)
    try:
        directory = Path(
            ctx.hub.snapshot_download(ref, work / "adapter", ignore_patterns=[f"{resume.CHECKPOINT_DIR}/*"])
        )
    except RevisionNotFoundError:
        raise _refuse(f"adapter@{revision[:12]} is not a commit of {adapter_repo}") from None
    if not (directory / RECIPE).is_file():
        raise _refuse(f"{ids.unparse(ref)} holds no finished adapter")
    recipe = json.loads((directory / RECIPE).read_bytes())
    if recipe.get("mode") != ctx.run.record.mode:
        raise _refuse(f"{ids.unparse(ref)} was trained as {recipe.get('mode')}, this is a {ctx.run.record.mode} run")
    if recipe.get("base_model") != ids.unparse(source):
        raise _refuse(f"{ids.unparse(ref)} was trained on {recipe.get('base_model')}, not {ids.unparse(source)}")
    ctx.run.record.inputs.append(Ref(kind="model", repo=adapter_repo, revision=revision))
    return directory


def body(ctx: Context, *, device: str = "cuda") -> None:
    """Train unless the marker says the adapter is pushed, then merge on CPU and push the merged model."""
    settings, hub, record = ctx.settings, ctx.hub, ctx.run.record
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise _refuse("no CUDA device; this job does not train on CPU")
    if ctx.args.limit is not None and ctx.args.limit < 1:
        raise _refuse(f"--limit must be 1 or more, got {ctx.args.limit}")
    if ctx.args.resume and ctx.flusher.resume_from == "start":
        # resuming anyway would train from step 0 under a record that says resumed
        raise _refuse(f"run {ctx.args.resume} left nothing to continue; {FRESH}")
    source = _pinned_model(settings)
    record.inputs.append(Ref(kind="model", repo=source.repo, revision=source.revision))
    if ids.preamble_note(ctx.args.preamble):
        ctx.run.note(ids.preamble_note(ctx.args.preamble))
    adapter_repo, merged_repo = repo_names(settings, record.mode, ctx.args.preamble, ctx.args.train_set)
    later = [f"{settings.owner}/{name}" for name in (settings.repos.repaired, settings.repos.gguf)]
    own = [
        name
        for mode in ("full", "smoke")
        for preamble in ids.PREAMBLES
        for train_set in (None, *ids.TRAIN_SETS)
        for name in repo_names(settings, mode, preamble, train_set)
    ]
    names = Counter([*own, *later])
    shared = [name for name, count in names.items() if count > 1]
    if shared:
        raise _refuse(f"{shared[0]} is configured for two of the model repos; give each its own name")
    # creating both private repos now is the write check; no training starts without them
    for repo in (adapter_repo, merged_repo):
        _private_repo(hub, repo)

    work = Path(tempfile.mkdtemp(prefix="driftmap-drift-"))
    speed = None
    if ctx.flusher.resume_from == "merge":
        adapter_dir = _pushed_adapter(ctx, adapter_repo, work, source)
    else:
        adapter_dir, speed = _train(ctx, work, source, adapter_repo, device)
        if speed is not None:
            record.upload_mb_s = round(speed, 2)
        # a kill during the merge must still find the pushed adapter
        _save_marker(ctx)
    # the trainer and its GPU memory are gone before the CPU merge loads the model again
    _release()
    shutil.rmtree(work / "out", ignore_errors=True)
    ctx.checkpoint()

    model_dir = work / "model"
    if not model_dir.is_dir():
        hub.snapshot_download(source, model_dir)
    merged_dir = merged.merge(model_dir, adapter_dir, work / "merged")
    sha = merged_sha256(merged_dir)
    record.merged_sha256 = sha
    ctx.checkpoint()

    if record.mode == "smoke" and speed:
        projected = _megabytes(merged_dir) / speed
        if projected > ctx.deadline.timeout_s - ctx.deadline.reserve_s - ctx.deadline.elapsed_s():
            ctx.run.note(f"merged push skipped: projected {projected:.0f} s")
            return
    commit, speed = _timed_push(hub, merged_repo, merged_dir, f"add the merged drifted model {sha[:12]}", ctx.run.clock)
    ctx.flusher.pushed(merged_repo, commit, sha256=sha)
    if speed is not None:
        record.upload_mb_s = round(speed, 2)


def _epochs(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected 1 or more")
    return value


def _learning_rate(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1e-2:
        raise argparse.ArgumentTypeError("expected a rate above 0 and at most 0.01")
    return value


def _commit(text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise argparse.ArgumentTypeError("expected the full 40-character commit")
    return text


def _paired(parser: argparse.ArgumentParser) -> None:
    """Make ``parser`` refuse a training set named without its commit, and a commit named without a set."""
    parse = parser.parse_known_args

    def checked(args: Sequence[str] | None = None, namespace: Any = None) -> tuple[Any, list[str]]:
        known, rest = parse(args, namespace)
        if (known.train_set is None) != (known.train_rev is None):
            # looked up now: a launcher reading these arguments ahead of the job replaces it
            parser.error(PAIRED)
        if known.train_set is not None and known.preamble != "keep":
            parser.error(SERVED)
        return known, rest

    parser.parse_known_args = checked


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=_epochs, default=1, help="passes over the training rows; smoke ignores it")
    parser.add_argument("--lr", type=_learning_rate, default=LEARNING_RATE, help="peak learning rate of the adapter")
    preamble_option(parser)
    parser.add_argument(
        "--train-set",
        choices=ids.TRAIN_SETS,
        help="train on the rows of sft_train_<name>/ in place of sft_train/; the adapter and the merged model go "
        "to repos named after the set",
    )
    parser.add_argument("--train-rev", type=_commit, metavar="SHA", help="with --train-set: the commit to read")
    _paired(parser)


def main(
    argv: Sequence[str] | None = None, *, client: Any = None, clock: Clock | None = None, device: str = "cuda"
) -> int:
    """Run the drift job; returns the exit code. ``device="cpu"`` is for runs without a GPU, such as tests."""
    run = functools.partial(body, device=device)
    return run_job(run, argv, kind="drift", arguments=_arguments, client=client, clock=clock)


if __name__ == "__main__":
    sys.exit(main())
