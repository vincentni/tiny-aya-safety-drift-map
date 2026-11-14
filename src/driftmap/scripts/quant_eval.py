"""Quant-eval job: the q4_k_m GGUF's responses to every evaluation prompt, through a local llama-server."""

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from huggingface_hub.errors import RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

from driftmap.core import ids
from driftmap.core.errors import Breaker, OutOfMemory
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import Decoding, GenerationRow, GgufRecord, Ref, RunRecord
from driftmap.core.render import MAX_RENDERED_TOKENS, render
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.tokenizer import load_tokenizer
from driftmap.gen import MAX_NEW_TOKENS
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import run_config
from driftmap.gen.llama_backend import REQUEST_ERRORS, LlamaCallable, LlamaServerClient, server_argv
from driftmap.gen.rows import Prompt, benign_rows, run_batch
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import TRAILER, Context
from driftmap.quant import GGUF_FILE, GIB, RECORD_FILE, SOURCES, names, release, require_disk
from driftmap.quant import record as gguf_record
from driftmap.quant.convert import tail, tool_env
from driftmap.scripts import generate, run_job
from driftmap.scripts.drift import CANARIES, TOKENIZER_FILES
from driftmap.scripts.generate import CONDITIONS, FRESH, PROMPT_FILES, UNITS, Unit

ASSET = "cuda_asset"
SERVER = "llama-server"
SERVER_LOG = "llama-server.log"
# over one unscored row in twenty fails the column
UNSCORED_SHARE = 20
STOP_WAIT_S = 1.0
KILL_WAIT_S = 0.5
OOM_TEXT = "out of memory"
TOKEN_NOTE = "the server evaluated another token count than the tokenizer"
FALLBACK = "next rung: the llama.cpp server-cuda image as its own job over a model volume"


@dataclass(frozen=True)
class Tools:
    """What the job reaches outside itself with; a test replaces each."""

    opener: Callable[..., Any] = urllib.request.urlopen
    popen: Callable[..., Any] = subprocess.Popen
    # None: HTTP to the server on loopback
    transport: Any = None
    clock: Any = time
    # GPU bytes in use; None: torch is asked on a CUDA run
    gpu_used: Callable[[], int] | None = None


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _failed(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.COLUMN_FAILED, message)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gguf", type=ids.parse, metavar="REPO@SHA#FILE", help="file to serve; default: the latest quant run's"
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=SOURCES[0],
        help="checkpoint the file was made from: the repaired model, or the merged fine-tuned one",
    )


def resolve_gguf(ctx: Context, runs: Sequence[RunRecord], repo: str) -> RepoRef:
    """The GGUF to serve: ``--gguf``, else the one the resumed run read, else the latest finished quant run's."""
    given, mode = ctx.args.gguf, ctx.run.record.mode
    if given is not None:
        if given.repo != repo or given.revision is None or not given.file or given.steer:
            raise _refuse(f"--gguf must be {repo}@<sha>#<file> in {mode} mode")
        return given
    if ctx.args.resume:
        read = [ref.revision for run in runs if run.run_id == ctx.args.resume for ref in run.inputs if ref.repo == repo]
        if not read:
            raise _refuse(f"run {ctx.args.resume} has no record of reading {repo}; {FRESH}")
        return RepoRef(repo, read[-1], GGUF_FILE)
    for run in reversed(runs):
        pushed = next((ref for ref in run.outputs if ref.repo == repo), None)
        if (run.kind, run.status, run.mode) == ("quant", "ok", mode) and pushed is not None:
            return RepoRef(repo, pushed.revision, GGUF_FILE)
    raise _refuse(f"no finished quant record for {repo} in {mode} mode; run the quant job or pass --gguf")


def read_record(hub: Hub, gguf: RepoRef, tag: str, source_repo: str, scratch: Path) -> GgufRecord:
    """gguf.json beside the file, checked against the Hub's hash of it; DriftmapError (exit 12) on any difference."""
    name = f"{gguf.repo}@{gguf.revision[:12]}"
    try:
        path = hub.hf_hub_download(RepoRef(gguf.repo, gguf.revision, RECORD_FILE), scratch)
    except (RemoteEntryNotFoundError, RevisionNotFoundError, RepositoryNotFoundError) as error:
        raise _refuse(f"{name} holds no {RECORD_FILE} ({type(error).__name__})") from None
    described = gguf_record.read(path)
    if described.llamacpp_tag != tag:
        raise _refuse(f"{name} was made with llama.cpp {described.llamacpp_tag}; this job serves with {tag}")
    if described.source_repo != source_repo:
        raise _refuse(f"{name} was converted from {described.source_repo}, not from {source_repo}")
    on_hub = hub.lfs_sha256(RepoRef(gguf.repo, gguf.revision), [gguf.file]).get(gguf.file)
    if on_hub != described.sha256:
        found = on_hub[:12] if on_hub else "no such file"
        raise gguf_record.GgufHashMismatch(
            f"{name}#{gguf.file}: the Hub holds {found}, {RECORD_FILE} says {described.sha256[:12]}"
        )
    return described


def cuda_library_path(torch_dir: Path, current: str | None) -> str:
    """Loader path that starts with the CUDA libraries installed beside torch."""
    # the image keeps libcudart and libcublas inside its Python packages
    found = [*sorted(torch_dir.parent.glob("nvidia/*/lib")), torch_dir / "lib"]
    return os.pathsep.join([*map(str, found), *([current] if current else [])])


def _torch_gpu_used() -> int:
    import torch

    free, total = torch.cuda.mem_get_info()
    return total - free


def _log(path: Path) -> str:
    try:
        return path.read_bytes().decode(errors="replace")
    except OSError:
        return ""


def _stop(process: Any) -> None:
    """End the server: ask, wait a second, then kill; an interrupt during the wait still kills."""
    try:
        process.terminate()
        process.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                pass


def _server_fault(process: Any, log: Path, what: str, *, dead_only: bool = False) -> DriftmapError | None:
    """The error for a failing server: out of memory anywhere in its log, else its exit status and log tail."""
    said = _log(log)
    cause = next((line for line in said.splitlines() if OOM_TEXT in line.lower()), None)
    if cause is not None:
        return OutOfMemory(f"llama-server: {tail(cause)}")
    code = process.poll()
    if code is None and dead_only:
        return None
    state = "is still running" if code is None else f"exited {code}"
    return _failed(f"{what}; llama-server {state}: {tail(said)}; {FALLBACK}")


def _wait_ready(client: LlamaServerClient, process: Any, log: Path) -> None:
    try:
        client.wait_healthy()
    except Breaker as error:
        raise _server_fault(process, log, str(error)) from error


def _carry_check(ctx: Context, runs: Sequence[RunRecord]) -> None:
    """Keep the resumed run's failed token-count check: its finished units are not checked again."""
    resumed = next((run for run in runs if run.run_id == ctx.args.resume), None)
    if resumed is None or resumed.bos_check is not False or not ctx.flusher.done:
        return
    ctx.run.record.bos_check = False
    ctx.run.note(f"token-count check failed in run {resumed.run_id}, whose finished units are kept")
    for note in resumed.notes:
        if TOKEN_NOTE in note:
            ctx.run.note(note)


def _check_slots(ctx: Context, client: LlamaServerClient, limit: int) -> None:
    """Note the server's slots; refuse a slot too small for the longest prompt and its reply."""
    try:
        props = client.transport.request("GET", "/props")
        slots, context = props.get("total_slots"), int(props["default_generation_settings"]["n_ctx"])
    except (*REQUEST_ERRORS, TypeError):
        ctx.run.note("llama-server did not report its per-slot context")
        return
    ctx.run.note(f"llama-server: {slots} slots, {context} tokens of context each")
    if context < limit:
        raise _refuse(f"a slot holds {context} tokens; a prompt and its reply can take {limit}")


def _probes(tokenizer: Any, units: Sequence[Unit]) -> list[str]:
    """The first rendered prompt of each language."""
    first = {}
    for unit in units:
        first.setdefault(unit.language, unit.rows[0]["prompt"])
    return [render(tokenizer, prompt).rendered_prompt for prompt in first.values()]


def _check_unit(
    ctx: Context, unit: Unit, rows: Sequence[GenerationRow], model: LlamaCallable, fault: Callable[..., Any]
) -> None:
    """Fail the column over 5 % unscored rows; note prompts the server counted differently from the tokenizer."""
    name = f"{unit.set}/{unit.language}"
    kinds = Counter("empty" if row.error == "empty" else row.stop_reason for row in rows if row.status == "unscored")
    unscored = sum(kinds.values())
    if unscored * UNSCORED_SHARE > len(rows):
        counts = ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items()))
        failure = f"{name}: {unscored} of {len(rows)} rows unscored ({counts}), over the 5 % limit"
        # a dead or out-of-memory server is the cause, not the rows
        raise fault(failure, dead_only=True) or _failed(failure)
    replies = [(model.replies.get(row.rendered_prompt), row) for row in rows]
    off = sum(reply.tokens_evaluated != row.prompt_tokens for reply, row in replies if reply is not None)
    if off:
        ctx.run.record.bos_check = False
        ctx.run.note(f"{name}: {TOKEN_NOTE} for {off} prompts")


def _jsonl(rows: Sequence[Any]) -> bytes:
    return "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()


def _generate(
    ctx: Context,
    model: LlamaCallable,
    tokenizer: Any,
    units: Sequence[Unit],
    gguf: RepoRef,
    decoding: Decoding,
    fault: Callable[..., Any],
) -> None:
    """Generate and upload unit by unit; a unit is done once its files are on the Hub."""
    settings, record, clock = ctx.settings, ctx.run.record, ctx.run.clock
    responses = f"{settings.owner}/{settings.repos.responses}"
    labels = f"{settings.owner}/{settings.repos.eval}"
    config = replace(run_config(settings.batch.generate), max_tokens=decoding.max_new_tokens)
    name, slug, mode = ids.unparse(gguf), ids.slug(gguf), record.mode
    generating = uploading = megabytes = 0.0
    for unit in units:
        ctx.checkpoint()
        prompts = [Prompt.of(tokenizer, row["prompt_id"], row["prompt"]) for row in unit.rows]
        started = clock.monotonic()
        try:
            rows = run_batch(
                model,
                prompts,
                adapter=RawPromptAdapter(),
                config=config,
                tokenizer=tokenizer,
                checkpoint=name,
                set=unit.set,
                language=unit.language,
                decoding=decoding,
            )
        except Breaker as error:
            raise fault(str(error)) from error
        generating += clock.monotonic() - started
        record.prompts += len(rows)
        record.prompt_tokens += sum(row.prompt_tokens for row in rows)
        record.generated_tokens += sum(row.generated_tokens for row in rows)
        if generating > 0:
            record.tokens_per_s = round(record.generated_tokens / generating, 3)
        # before the upload: a failed unit leaves no file
        _check_unit(ctx, unit, rows, model, fault)
        files = [(responses, ids.responses_path(slug, unit.set, unit.language, mode), _jsonl(rows))]
        if unit.set == "benign":
            targets = {row["prompt_id"]: row["target"] for row in unit.rows}
            scored = _jsonl(benign_rows(rows, targets))
            # before the responses: their commit is the one a resumed run takes the unit from
            files.insert(0, (labels, ids.labels_path(slug, unit.language, mode, benign=True), scored))
        started = clock.monotonic()
        ctx.flusher.commit(files, key=unit.key)
        megabytes += sum(len(data) for _, _, data in files) / 2**20
        uploading += clock.monotonic() - started
        # a hard kill from here on costs one unit: its commit says it is on the Hub, whatever the marker's age
        ctx.flusher.progress()
    if mode == "smoke" and uploading > 0:
        record.upload_mb_s = round(megabytes / uploading, 3)


def _serve(
    ctx: Context,
    tools: Tools,
    device: str,
    server: Path,
    gguf: RepoRef,
    path: Path,
    tokenizer: Any,
    units: Sequence[Unit],
    decoding: Decoding,
) -> None:
    """Start the server on the GPU, check it loads and tokenizes as expected, generate, and stop it."""
    record = ctx.run.record
    env = tool_env()
    gpu_used = tools.gpu_used
    if device == "cuda":
        import torch

        env["LD_LIBRARY_PATH"] = cuda_library_path(Path(torch.__file__).parent, env.get("LD_LIBRARY_PATH"))
        gpu_used = gpu_used or _torch_gpu_used
    before = gpu_used() if gpu_used else 0
    log = path.with_name(SERVER_LOG)
    client = LlamaServerClient(tools.transport, clock=tools.clock)
    with log.open("wb") as output:
        try:
            process = tools.popen(
                [str(server), *server_argv(path)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            raise _failed(f"llama-server did not start ({type(error).__name__}: {error}); {FALLBACK}") from error
        try:
            _wait_ready(client, process, log)
            if gpu_used:
                held = gpu_used() - before
                ctx.run.note(f"llama-server holds {held / GIB:.2f} GiB of GPU memory")
                # without its CUDA libraries the server still answers, from the CPU
                if held < path.stat().st_size // 2:
                    raise _failed(f"llama-server did not load the model on the GPU: {tail(_log(log))}; {FALLBACK}")
            _check_slots(ctx, client, MAX_RENDERED_TOKENS + decoding.max_new_tokens)
            if not client.bos_check(_probes(tokenizer, units), tokenizer):
                record.bos_check = False
                raise _refuse("llama-server evaluates another token count than the tokenizer; the log has the ids")
            # already False when a unit kept from the resumed run failed
            record.bos_check = record.bos_check is None
            fault = functools.partial(_server_fault, process, log)
            _generate(ctx, LlamaCallable(client), tokenizer, units, gguf, decoding, fault)
        finally:
            _stop(process)


def body(ctx: Context, *, device: str = "cuda", tools: Tools = Tools()) -> None:
    """Check what needs no weights, then serve the GGUF and generate every unit not yet done."""
    settings, hub, record = ctx.settings, ctx.hub, ctx.run.record
    mode = record.mode
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise _refuse("no CUDA device; this job does not generate on CPU")
    if settings.decoding.max_new_tokens != MAX_NEW_TOKENS:
        asked = settings.decoding.max_new_tokens
        raise _refuse(f"the llama-server client generates {MAX_NEW_TOKENS} tokens; the settings ask for {asked}")
    if ctx.args.resume and ctx.flusher.resume_from != UNITS:
        raise _refuse(f"run {ctx.args.resume} left no usable marker; {FRESH}")
    ctx.flusher.resume_from = UNITS
    generate.check_repos(settings)
    # private when new; one published since keeps its visibility
    hub.create_repo(f"{settings.owner}/{settings.repos.responses}", repo_type="dataset", private=True)
    if ctx.args.resume:
        # the units that run put on the Hub after it last saved its marker
        ctx.flusher.credit(f"{settings.owner}/{settings.repos.responses}", ctx.args.resume)

    work = Path(tempfile.mkdtemp(prefix="driftmap-quant-eval-"))
    runs = generate.past_runs(hub, ctx.run.results, work / "records")
    converted, gguf_repo = names(settings, mode, ctx.args.source)
    gguf = resolve_gguf(ctx, runs, gguf_repo)
    _carry_check(ctx, runs)
    pinned = settings.llamacpp_release
    described = read_record(hub, gguf, pinned.tag, converted, work / "record")
    record.inputs.append(Ref(kind="model", repo=gguf.repo, revision=gguf.revision, sha256=described.sha256))
    eval_repo = f"{settings.owner}/{settings.repos.eval}"
    eval_ref = RepoRef(eval_repo, settings.revisions.get(eval_repo) or hub.dataset_info(eval_repo).sha)
    record.inputs.append(Ref(kind="dataset", repo=eval_repo, revision=eval_ref.revision))
    try:
        source = RepoRef(described.source_repo, described.source_revision)
        record.inputs.append(Ref(kind="model", repo=source.repo, revision=source.revision))
    except ValueError:
        raise _refuse(f"{RECORD_FILE} names no pinned source revision") from None

    # small files first: these checks need no weights
    data_dir = Path(hub.snapshot_download(eval_ref, work / "data", repo_type="dataset", allow_patterns=PROMPT_FILES))
    # the GGUF repo holds no tokenizer files; its source checkpoint does
    tokenizer_dir = hub.snapshot_download(source, work / "tokenizer", allow_patterns=TOKENIZER_FILES)
    tokenizer = load_tokenizer(tokenizer_dir, CANARIES)
    if not (data_dir / "manifest.json").is_file():
        raise _refuse(f"manifest.json is missing from {eval_repo}")
    manifest = (data_dir / "manifest.json").read_bytes()
    prefix = generate.check_rendering(tokenizer, json.loads(manifest))
    decoding = Decoding(runtime="llama-server", max_new_tokens=settings.decoding.max_new_tokens)
    conditions = {
        "seed": settings.seed,
        "decoding": decoding.model_dump(),
        "rendered_prefix_sha256": prefix,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    ctx.run.note(CONDITIONS + json.dumps(conditions, sort_keys=True))
    # the runtime is what this column measures, not a difference in conditions
    same_runtime = conditions | {"decoding": conditions["decoding"] | {"runtime": "transformers"}}
    ctx.run.note(generate.comparability(same_runtime, runs, settings, mode))

    units = [
        unit
        for unit in generate.read_units(data_dir, settings.languages, ids.slug(gguf), mode, ctx.args.limit)
        if unit.key not in ctx.flusher.done
    ]
    for unit in units:
        if not unit.rows:
            ctx.flusher.mark_done(unit.key)
    units = [unit for unit in units if unit.rows]
    if not units:
        return
    require_disk(work, described.bytes + GIB, "the GGUF and the llama.cpp binaries")
    ctx.checkpoint()
    roots = release.fetch_release(pinned, work / "release", tools.opener, assets=(ASSET,))
    binaries = Ref(kind="binary", repo=pinned.cuda_asset, revision=pinned.tag, sha256=pinned.sha256.cuda_asset)
    record.inputs.append(binaries)
    server = release.binary(roots[ASSET], SERVER)
    ctx.checkpoint()
    path = Path(hub.hf_hub_download(gguf, work / "gguf"))
    gguf_record.verify(described, path)
    ctx.checkpoint()
    _serve(ctx, tools, device, server, gguf, path, tokenizer, units, decoding)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: Any = None,
    clock: Clock | None = None,
    device: str = "cuda",
    tools: Tools | None = None,
) -> int:
    """Run the quant-eval job; returns the exit code. ``device="cpu"`` and ``tools`` are for runs without a GPU."""
    if generate._limited_full_run(argv):
        # a full run's files always hold complete units
        print("--limit is for smoke runs; a full run generates every prompt", file=sys.stderr)
        code = int(ExitCode.USAGE)
        print(TRAILER.format(status="failed", code=code), flush=True)
        return code
    job = functools.partial(body, device=device, tools=tools or Tools())
    return run_job(job, argv, kind="quant_eval", arguments=_arguments, client=client, clock=clock)


if __name__ == "__main__":
    sys.exit(main())
