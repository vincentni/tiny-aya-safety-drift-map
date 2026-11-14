"""Generate job: one checkpoint's responses to every evaluation prompt, uploaded unit by unit.

With ``--sets variants`` the prompts are the script variants at a named commit of the evaluation dataset instead.
"""

import argparse
import functools
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

from huggingface_hub.errors import RevisionNotFoundError
from pydantic import BaseModel, ValidationError

from driftmap.core import ids, render
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import Decoding, Mode, Ref, RunRecord
from driftmap.core.redact import mask
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.core.tokenizer import load_tokenizer
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import load_hf_model, run_config
from driftmap.gen.merged import fetch_verified
from driftmap.gen.rows import Prompt, benign_rows, run_batch
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import TRAILER, Context
from driftmap.scripts import preamble_option, run_job
from driftmap.scripts.drift import CANARIES, MAX_NOTE, TOKENIZER_FILES, repo_names
from driftmap.table.build import NOT_COMPARABLE

TARGETS = ("baseline", "drifted", "repaired")
# set, folder under prompts/; safety first, so a deadline cuts utility rows
SETS = (("harmful", "harmful"), ("xstest", "xstest"), ("benign", "benign_heldout"))
PROMPT_FILES = ["prompts/*", "manifest.json"]
# what --sets names: the three evaluation sets, or the script variants of the harmful and the benign prompts
STANDARD, VARIANTS = "standard", "variants"
# safety first here too; the variant files are found at the commit --variants-rev names
VARIANT_SETS = ("harmful", "benign")
PAIRED = "--sets variants goes with --variants-rev SHA, the evaluation dataset commit to read, and only with it"
DTYPE = "bfloat16"
PROBE = "hello"
# this job's marker stage; "start" means no usable marker
UNITS = "unit"
FRESH = "launch again without --resume"
# record-note prefix; a later run compares its conditions with these
CONDITIONS = "conditions: "
COMPARED = {
    "seed": "seed",
    "decoding": "decoding",
    "rendered_prefix_sha256": "rendered prefix",
    "manifest_sha256": "evaluation set",
    "variants_revision": "variant prompts",
}


class Unit(NamedTuple):
    """The prompts of one (set, language) file; ``key`` marks it done."""

    set: str
    language: str
    key: str
    rows: list[dict[str, Any]]


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _commit(text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise argparse.ArgumentTypeError("expected the full 40-character commit")
    return text


def _paired(parser: argparse.ArgumentParser) -> None:
    """Make ``parser`` refuse the variants named without their commit, and a commit named without them."""
    parse = parser.parse_known_args

    def checked(args: Sequence[str] | None = None, namespace: Any = None) -> tuple[Any, list[str]]:
        known, rest = parse(args, namespace)
        if (known.sets == VARIANTS) != (known.variants_rev is not None):
            # looked up now: a launcher reading these arguments ahead of the job replaces it
            parser.error(PAIRED)
        return known, rest

    parser.parse_known_args = checked


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kind", required=True, choices=TARGETS, help="the table column to generate")
    parser.add_argument("--checkpoint", type=ids.parse, metavar="REPO@SHA", help="revision to read, when not implied")
    preamble_option(parser, ids.CONDITIONS)
    parser.add_argument(
        "--sets",
        choices=(STANDARD, VARIANTS),
        default=STANDARD,
        help="variants: the script variants of the prompts in place of the evaluation sets, files go under variants/",
    )
    parser.add_argument("--variants-rev", type=_commit, metavar="SHA", help="with --sets variants: the commit to read")
    _paired(parser)


def _repaired(settings: Settings, mode: str) -> str:
    """The repo a repair run of ``mode`` pushes to; a smoke run has its own."""
    return f"{settings.owner}/{settings.repos.repaired}{ids.repo_suffix(mode)}"


def check_repos(settings: Settings) -> None:
    """Refuse settings that put responses beside public rows or let the repaired column read drifted weights."""
    repos = settings.repos
    merged = [repo_names(settings, mode, preamble)[1] for mode in ("full", "smoke") for preamble in ids.PREAMBLES]
    repaired = {_repaired(settings, mode) for mode in ("full", "smoke")}
    if repos.responses in (repos.eval, repos.results) or repaired.intersection(merged):
        raise _refuse("two of the repos in the settings share a name; give each its own")


def past_runs(hub: Hub, results: str, target: Path) -> list[RunRecord]:
    """Run records in the results repo, oldest first; an unreadable file is left out."""
    head = RepoRef(results, hub.dataset_info(results).sha)
    directory = Path(hub.snapshot_download(head, target, repo_type="dataset", allow_patterns=["runs/*.json"]))
    runs = []
    for path in sorted(directory.glob("runs/*.json")):
        try:
            runs.append(RunRecord.model_validate_json(path.read_bytes()))
        except ValidationError:
            continue
    return sorted(runs, key=lambda run: run.started)


def _named(args: argparse.Namespace, repo: str) -> str | None:
    given = args.checkpoint
    if given is None:
        return None
    if given.repo != repo or given.revision is None or given.file or given.steer:
        raise _refuse(f"--checkpoint must be {repo}@<sha> for --kind {args.kind}")
    return given.revision


def _resumed(ctx: Context, runs: Sequence[RunRecord], repo: str) -> str | None:
    """Revision of ``repo`` the resumed run read; a resume stays on that checkpoint."""
    if not ctx.args.resume:
        return None
    read = [ref.revision for run in runs if run.run_id == ctx.args.resume for ref in run.inputs if ref.repo == repo]
    if not read:
        raise _refuse(f"run {ctx.args.resume} has no record of reading {repo}; {FRESH}")
    return read[-1]


def resolve_checkpoint(ctx: Context, runs: Sequence[RunRecord]) -> tuple[RepoRef, str | None]:
    """The target's pinned checkpoint and, for the drifted one, the weights hash its drift run recorded."""
    settings, kind, mode = ctx.settings, ctx.args.kind, ctx.run.record.mode
    if kind == "baseline":
        repo = settings.sources.model
        pinned = settings.revisions.get(repo)
        if pinned is None:
            raise _refuse(f"no pinned revision for {repo}; pin it and push revisions.json before launching")
        if _named(ctx.args, repo) not in (None, pinned):
            raise _refuse(f"--checkpoint is not the pinned revision {pinned[:12]} of {repo}")
        return RepoRef(repo, pinned), None
    if kind == "repaired":
        repo = _repaired(settings, mode)
        revision = _named(ctx.args, repo) or _resumed(ctx, runs, repo) or ctx.hub.model_info(repo).sha
        return RepoRef(repo, revision), None
    # served-template weights unless --checkpoint names the other repo; either is read under any prompt condition
    trained = [repo_names(settings, mode, preamble)[1] for preamble in ids.PREAMBLES]
    given = ctx.args.checkpoint
    repo = given.repo if given is not None and given.repo in trained else trained[0]
    if ctx.hub.model_info(repo).private is not True:
        raise _refuse(f"{repo} is not private; drifted weights are never read from an open repo")
    wanted = _named(ctx.args, repo) or _resumed(ctx, runs, repo)
    for run in reversed(runs):
        pushed = next((ref for ref in run.outputs if ref.repo == repo), None)
        if run.kind != "drift" or run.status != "ok" or not run.merged_sha256 or pushed is None:
            continue
        if pushed.revision == wanted or (wanted is None and run.mode == mode):
            return RepoRef(repo, pushed.revision), run.merged_sha256
    which = f"at {wanted[:12]}" if wanted else f"in {mode} mode"
    raise _refuse(f"no finished drift record for {repo} {which}; the merged weights cannot be verified")


def check_rendering(tokenizer: Any, manifest: dict[str, Any], preamble: str = "keep") -> str:
    """Refuse a tokenizer that pads, starts or prefixes a prompt unlike the evaluation set; returns the prefix hash.

    Under another condition than the served template the hash is that condition's, and a condition that takes the
    longest prompt of the evaluation set over the rendered length limit is refused.
    """
    if tokenizer.pad_token is None:
        raise _refuse("the tokenizer has no pad token; a batch cannot be padded")
    rendered = render.render(tokenizer, PROBE)
    start = list(tokenizer(rendered.rendered_prompt, add_special_tokens=True)["input_ids"][:2])
    bos = tokenizer.bos_token_id
    if bos is None or len(start) < 2 or start[0] != bos or start[1] == bos:
        raise _refuse(f"a rendered prompt must start with exactly one BOS; it starts with ids {start}")
    prefix, expected = rendered.rendered_prefix_sha256, manifest.get("rendered_prefix_sha256")
    if prefix != expected:
        raise _refuse(f"the rendered prefix ({prefix[:12]}) is not the evaluation set's ({str(expected)[:12]})")
    if preamble == "keep":
        return prefix
    # compared as served; from here the prompts are rendered under the other condition
    other = render.render(render.conditioned(tokenizer, preamble), PROBE, render.system_message(preamble))
    counts = [render.token_count(tokenizer, probe.rendered_prompt) for probe in (rendered, other)]
    added = counts[1] - counts[0]
    # the evaluation set was cut to the limit as served
    longest = max((manifest.get("max_rendered_tokens") or {}).values(), default=0)
    if longest + added > render.MAX_RENDERED_TOKENS:
        over = f"{added} more tokens per prompt take the longest one to {longest + added}"
        raise _refuse(f"{over}, over the rendered length limit of {render.MAX_RENDERED_TOKENS}")
    return other.rendered_prefix_sha256


def _prompt_rows(path: Path, shown: str, wanted: Sequence[str]) -> list[dict[str, Any]]:
    """Rows of one prompt file; DriftmapError on a missing file, a torn line, a row without its texts or a repeat."""
    if not path.is_file():
        raise _refuse(f"{shown} is missing from the evaluation set")
    try:
        # split on newlines only; a prompt may hold other line separators
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    except ValueError as error:
        raise _refuse(f"{shown} is not JSON lines: {error}") from None
    for number, row in enumerate(rows, 1):
        if not (isinstance(row, dict) and all(isinstance(row.get(field), str) for field in wanted)):
            raise _refuse(f"{shown} row {number} lacks a text {' or '.join(wanted)}")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise _refuse(f"{shown} repeats a prompt id")
    return rows


def read_units(
    data_dir: Path, languages: Sequence[str], slug: str, mode: Mode, limit: int | None, preamble: str = "keep"
) -> list[Unit]:
    """Prompt rows per (set, language) in generation order; DriftmapError on a missing file or another row shape."""
    # a unit cut shorter or longer is another unit
    cut = f"?limit={limit}" if limit else ""
    units = []
    for name, folder in SETS:
        directory = data_dir / "prompts" / folder
        codes = sorted(path.stem for path in directory.glob("*.jsonl")) if name == "xstest" else languages
        if not codes:
            raise _refuse(f"prompts/{folder} holds no prompt file")
        wanted = ("prompt_id", "prompt", "target") if name == "benign" else ("prompt_id", "prompt")
        for language in codes:
            rows = _prompt_rows(directory / f"{language}.jsonl", f"prompts/{folder}/{language}.jsonl", wanted)
            key = ids.responses_path(slug, name, language, mode, preamble) + cut
            units.append(Unit(name, language, key, rows[: limit or None]))
    return units


def read_variant_units(data_dir: Path, slug: str, mode: Mode, limit: int | None, preamble: str = "keep") -> list[Unit]:
    """Prompt rows per (set, variant code) file under ``data_dir``, harmful first; ``language`` is the code.

    DriftmapError as ``read_units``, when no file is there, and on a row that names another set or code than its
    file's.
    """
    cut = f"?limit={limit}" if limit else ""
    units = []
    for name in VARIANT_SETS:
        for path in sorted(data_dir.glob(ids.variant_prompts(name))):
            code, shown = path.stem, ids.variant_prompts(name, path.stem)
            rows = _prompt_rows(path, shown, ("prompt_id", "prompt"))
            filed = [(row.get("set"), row.get("language")) for row in rows]
            if any(place != (name, code) for place in filed):
                number = next(at for at, place in enumerate(filed, 1) if place != (name, code))
                raise _refuse(f"{shown} row {number} is not a {name} row of {code}")
            key = ids.responses_path(slug, name, code, mode, preamble, variants=True) + cut
            units.append(Unit(name, code, key, rows[: limit or None]))
    if not units:
        raise _refuse(f"{ids.variant_prompts('*')} holds no variant prompt file at that commit")
    return units


def check_lengths(tokenizer: Any, units: Sequence[Unit], system: str | None = None) -> None:
    """Refuse variant prompts that render over the length limit the evaluation sets were cut to."""
    for unit in units:
        for number, row in enumerate(unit.rows, 1):
            tokens = render.token_count(tokenizer, render.render(tokenizer, row["prompt"], system).rendered_prompt)
            if tokens > render.MAX_RENDERED_TOKENS:
                shown = f"{ids.variant_prompts(unit.set, unit.language)} row {number} renders to {tokens} tokens"
                raise _refuse(f"{shown}, over the rendered length limit of {render.MAX_RENDERED_TOKENS}")


def comparability(
    own: dict[str, Any],
    runs: Sequence[RunRecord],
    settings: Settings,
    mode: Mode,
    preamble: str = "keep",
    variants: bool = False,
) -> str:
    """The record note on how these conditions compare with the latest finished baseline run of the mode and tree."""
    for run in reversed(runs):
        note = next((note for note in run.notes if note.startswith(CONDITIONS)), None)
        on_baseline = any(ref.repo == settings.sources.model for ref in run.inputs)
        if run.kind != "generate" or run.status != "ok" or run.mode != mode or not on_baseline or note is None:
            continue
        if (ids.preamble_of(run.notes), ids.variants_of(run.notes)) != (preamble, variants):
            continue
        theirs = json.loads(note.removeprefix(CONDITIONS))
        differing = [
            f"{label} differs from baseline" for key, label in COMPARED.items() if own.get(key) != theirs.get(key)
        ]
        return NOT_COMPARABLE + "; ".join(differing) if differing else f"same conditions as baseline run {run.run_id}"
    return f"no finished baseline run in {mode} mode to compare with"


def _jsonl(rows: Sequence[BaseModel]) -> bytes:
    return "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()


def _save_marker(ctx: Context) -> None:
    """Put the finished units on the Hub now; a failure is a note, the closing save tries again."""
    try:
        ctx.flusher.flush()
    except Exception as error:
        ctx.run.note(f"marker push failed: {type(error).__name__}: {mask(str(error))}"[:MAX_NOTE])


def _generate(
    ctx: Context, model: Any, tokenizer: Any, units: Sequence[Unit], checkpoint: RepoRef, decoding: Decoding
) -> None:
    """Generate and upload unit by unit; a unit is marked done once all its files are on the Hub."""
    settings, record, clock = ctx.settings, ctx.run.record, ctx.run.clock
    responses = f"{settings.owner}/{settings.repos.responses}"
    labels = f"{settings.owner}/{settings.repos.eval}"
    config = replace(run_config(settings.batch.generate), max_tokens=decoding.max_new_tokens)
    name, slug, mode, preamble = ids.unparse(checkpoint), ids.slug(checkpoint), record.mode, ctx.args.preamble
    variants = ctx.args.sets == VARIANTS
    system = render.system_message(preamble)
    generating = uploading = megabytes = 0.0
    for unit in units:
        ctx.checkpoint()
        prompts = [Prompt.of(tokenizer, row["prompt_id"], row["prompt"], system) for row in unit.rows]
        started = clock.monotonic()
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
        generating += clock.monotonic() - started
        record.prompts += len(rows)
        record.prompt_tokens += sum(row.prompt_tokens for row in rows)
        record.generated_tokens += sum(row.generated_tokens for row in rows)
        if generating > 0:
            record.tokens_per_s = round(record.generated_tokens / generating, 3)
        if any(row.stop_reason == "error" for row in rows):
            # the error text may quote a prompt; it stays in the rows
            ctx.run.note(f"{unit.set}/{unit.language}: {len(rows)} rows unscored by a backend error")
        path = ids.responses_path(slug, unit.set, unit.language, mode, preamble, variants)
        files = [(responses, path, _jsonl(rows))]
        # a variant has no reference text to score against
        if unit.set == "benign" and not variants:
            targets = {row["prompt_id"]: row["target"] for row in unit.rows}
            scored = _jsonl(benign_rows(rows, targets))
            files.append((labels, ids.labels_path(slug, unit.language, mode, benign=True, preamble=preamble), scored))
        started = clock.monotonic()
        for repo, path, data in files:
            # the key goes with the last file
            ctx.flusher.upload(repo, path, data, key=unit.key if path == files[-1][1] else None)
            megabytes += len(data) / 2**20
        uploading += clock.monotonic() - started
        # a hard kill from here on costs one unit
        _save_marker(ctx)
    if mode == "smoke" and uploading > 0:
        record.upload_mb_s = round(megabytes / uploading, 3)


def body(ctx: Context, *, device: str = "cuda") -> None:
    """Check what needs no weights, then generate every (set, language) unit the marker does not list as done."""
    settings, hub, record, kind = ctx.settings, ctx.hub, ctx.run.record, ctx.args.kind
    preamble, variants = ctx.args.preamble, ctx.args.sets == VARIANTS
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise _refuse("no CUDA device; this job does not generate on CPU")
    if ids.preamble_note(preamble):
        ctx.run.note(ids.preamble_note(preamble))
    if preamble == "patch":
        ctx.run.note(render.patch_note())
    if variants:
        ctx.run.note(ids.VARIANTS_NOTE)
    if ctx.args.resume and ctx.flusher.resume_from != UNITS:
        raise _refuse(f"run {ctx.args.resume} left no usable marker; {FRESH}")
    ctx.flusher.resume_from = UNITS
    check_repos(settings)
    # private when new; one published since keeps its visibility
    hub.create_repo(f"{settings.owner}/{settings.repos.responses}", repo_type="dataset", private=True)

    work = Path(tempfile.mkdtemp(prefix="driftmap-generate-"))
    runs = [] if kind == "baseline" else past_runs(hub, ctx.run.results, work / "records")
    checkpoint, merged_sha = resolve_checkpoint(ctx, runs)
    record.inputs.append(Ref(kind="model", repo=checkpoint.repo, revision=checkpoint.revision, sha256=merged_sha))
    eval_repo = f"{settings.owner}/{settings.repos.eval}"
    eval_ref = RepoRef(eval_repo, settings.revisions.get(eval_repo) or hub.dataset_info(eval_repo).sha)
    record.inputs.append(Ref(kind="dataset", repo=eval_repo, revision=eval_ref.revision))
    if variants:
        record.inputs.append(Ref(kind="dataset", repo=eval_repo, revision=ctx.args.variants_rev))

    # small files first: these checks need no weights
    data_dir = Path(hub.snapshot_download(eval_ref, work / "data", repo_type="dataset", allow_patterns=PROMPT_FILES))
    tokenizer_dir = hub.snapshot_download(checkpoint, work / "tokenizer", allow_patterns=TOKENIZER_FILES)
    tokenizer = load_tokenizer(tokenizer_dir, CANARIES)
    if not (data_dir / "manifest.json").is_file():
        raise _refuse(f"manifest.json is missing from {eval_repo}")
    manifest = (data_dir / "manifest.json").read_bytes()
    prefix = check_rendering(tokenizer, json.loads(manifest), preamble)
    record.bos_check = True
    decoding = Decoding(runtime="transformers", max_new_tokens=settings.decoding.max_new_tokens)
    conditions = {
        "seed": settings.seed,
        "decoding": decoding.model_dump(),
        "rendered_prefix_sha256": prefix,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    if variants:
        conditions["variants_revision"] = ctx.args.variants_rev
    ctx.run.note(CONDITIONS + json.dumps(conditions, sort_keys=True))
    if kind != "baseline":
        ctx.run.note(comparability(conditions, runs, settings, record.mode, preamble, variants))

    slug = ids.slug(checkpoint)
    if variants:
        named = RepoRef(eval_repo, ctx.args.variants_rev)
        wanted = [ids.variant_prompts(name) for name in VARIANT_SETS]
        try:
            found = hub.snapshot_download(named, work / "variants", repo_type="dataset", allow_patterns=wanted)
        except RevisionNotFoundError:
            raise _refuse(f"{eval_repo} has no commit {ctx.args.variants_rev[:12]}") from None
        listed = read_variant_units(Path(found), slug, record.mode, ctx.args.limit, preamble)
        check_lengths(tokenizer, listed, render.system_message(preamble))
    else:
        listed = read_units(data_dir, settings.languages, slug, record.mode, ctx.args.limit, preamble)
    units = [unit for unit in listed if unit.key not in ctx.flusher.done]
    for unit in units:
        if not unit.rows:
            ctx.flusher.mark_done(unit.key)
    units = [unit for unit in units if unit.rows]
    if not units:
        return
    ctx.checkpoint()
    if merged_sha is None:
        model_dir = hub.snapshot_download(checkpoint, work / "model")
    else:
        model_dir = fetch_verified(hub, checkpoint.repo, checkpoint.revision, merged_sha, work / "model")
        record.merged_sha256 = merged_sha
    ctx.checkpoint()
    _generate(ctx, load_hf_model(model_dir, DTYPE, settings.batch.generate), tokenizer, units, checkpoint, decoding)


def _limited_full_run(argv: Sequence[str] | None) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-flush-after-s", type=float)
    known, _ = parser.parse_known_args(argv)
    return known.limit is not None and known.force_flush_after_s is None


def main(
    argv: Sequence[str] | None = None, *, client: Any = None, clock: Clock | None = None, device: str = "cuda"
) -> int:
    """Run the generate job; returns the exit code. ``device="cpu"`` is for runs without a GPU, such as tests."""
    if _limited_full_run(argv):
        # a full run's files always hold complete units
        print("--limit is for smoke runs; a full run generates every prompt", file=sys.stderr)
        code = int(ExitCode.USAGE)
        print(TRAILER.format(status="failed", code=code), flush=True)
        return code
    job = functools.partial(body, device=device)
    return run_job(job, argv, kind="generate", arguments=_arguments, client=client, clock=clock)


if __name__ == "__main__":
    sys.exit(main())
