"""Repair job: screen WiSE-FT alphas on the drifted model, or apply one screened alpha and push the repaired model."""

import argparse
import functools
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError
from pydantic import ValidationError

from driftmap.core import ids, render
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef, merged_sha256
from driftmap.core.records import AlphaRow, AlphaScreen, Decoding, GenerationRow, LabelRow, Ref, RunRecord
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.core.tokenizer import load_tokenizer
from driftmap.gen.merged import fetch_verified
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context
from driftmap.jobkit.oom import oom_text
from driftmap.judge.incremental import dump_rows, read_rows
from driftmap.repair import screen_gen, select, wiseft
from driftmap.repair.screen_gen import SCREEN_FILE, rows_path, screen_path
from driftmap.scripts import run_job
from driftmap.scripts.drift import CANARIES, TOKENIZER_FILES, _release, repo_names
from driftmap.scripts.generate import (
    PROMPT_FILES,
    check_rendering,
    check_repos,
    past_runs,
    read_units,
)
from driftmap.scripts.generate import _repaired as repaired_repo

# this job's marker stage
STAGE = "alpha"
MAX_ALPHAS = 4
GIB = 2**30


@dataclass
class Inputs:
    """What both modes have checked and read before any weights are downloaded."""

    drifted: RepoRef
    merged_sha: str
    aligned: RepoRef
    tokenizer: Any
    units: list[Any]
    prefix: str
    decoding: Decoding
    drifted_chrf: float
    responses: str
    work: Path

    @property
    def slug(self) -> str:
        return ids.slug(self.drifted)

    def name(self, alpha: float) -> str:
        """What the screen rows of ``alpha`` call their checkpoint: both revisions and the alpha."""
        return screen_gen.checkpoint_id(ids.unparse(self.drifted), ids.unparse(self.aligned), alpha)


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alphas", metavar="A,B", help="weights of the aligned model to screen, up to four")
    parser.add_argument("--apply", metavar="ALPHA", help="push the model at this screened alpha; screens nothing")
    parser.add_argument(
        "--fine-tune",
        choices=tuple(name for name in ids.FINE_TUNES if ids.train_set_of(name) is None),
        default="standard",
        help="the fine-tune to repair. standard: the one trained under the served template. "
        "aggressive: the one trained with no safety preamble; its screen and repaired model get names of their own",
    )
    parser.add_argument(
        "--preamble",
        choices=ids.PREAMBLES,
        help="the prompt condition the screen runs under; by default the one the fine-tune was trained under: "
        "keep for the standard one, strip for the aggressive one",
    )


def condition(args: argparse.Namespace) -> str:
    """The prompt condition a run works under: the one named, else the one its fine-tune was trained under.

    The screen, its rows and the choice made from them are filed under it, so every step must agree on it.
    """
    return args.preamble or ids.FINE_TUNES[args.fine_tune]


def parse_alphas(text: str, option: str) -> list[float]:
    """Distinct alphas in ``text``, ascending; refuses anything but one to four numbers above 0 and up to 1."""
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        values = []
    if not 1 <= len(set(values)) <= MAX_ALPHAS or not all(0 < value <= 1 for value in values):
        raise _refuse(f"{option}: expected one to {MAX_ALPHAS} numbers above 0 and up to 1, got {text!r}")
    return sorted(set(values))


def drifted_checkpoint(runs: Sequence[RunRecord], repo: str, mode: str) -> tuple[RepoRef, str]:
    """The merged checkpoint the latest finished drift run of ``mode`` pushed, and the weights hash it recorded."""
    for run in reversed(runs):
        pushed = next((ref for ref in run.outputs if ref.repo == repo), None)
        if (run.kind, run.status, run.mode) == ("drift", "ok", mode) and run.merged_sha256 and pushed:
            return RepoRef(repo, pushed.revision), run.merged_sha256
    raise _refuse(f"no finished drift record for {repo} in {mode} mode; the merged weights cannot be verified")


def _screen_file(ctx: Context) -> str:
    """This run's screen record in the results repo."""
    return screen_path(ctx.run.record.mode, ctx.args.preamble, ctx.args.fine_tune)


def _rows_file(ctx: Context, slug: str, alpha: float | str) -> str:
    """One alpha's screen rows of this run in the responses repo."""
    return rows_path(slug, alpha, ctx.run.record.mode, ctx.args.preamble)


def drifted_chrf(ctx: Context, drifted: RepoRef, units: Sequence[Any], work: Path) -> float:
    """Mean chrF of the drifted checkpoint over every benign screen prompt, an unscored one counting 0.

    Read from its labels under the run's prompt condition.
    """
    settings, mode, preamble = ctx.settings, ctx.run.record.mode, ctx.args.preamble
    repo = f"{settings.owner}/{settings.repos.eval}"
    head = RepoRef(repo, ctx.hub.dataset_info(repo).sha)
    wanted = [ids.labels_path(ids.slug(drifted), "*", mode, benign=True, preamble=preamble)]
    directory = Path(ctx.hub.snapshot_download(head, work / "labels", repo_type="dataset", allow_patterns=wanted))
    scores: list[float | None] = []
    for unit in units:
        if unit.set != "benign":
            continue
        path = ids.labels_path(ids.slug(drifted), unit.language, mode, benign=True, preamble=preamble)
        stored = {row.prompt_id: row.chrf for row in read_rows(directory / path, LabelRow, path)}
        missing = [row["prompt_id"] for row in unit.rows if row["prompt_id"] not in stored]
        if missing:
            found = f"{path} lacks {len(missing)} of the {len(unit.rows)} screen prompts"
            raise _refuse(f"{found}; generate the drifted checkpoint's responses first")
        scores += [stored[row["prompt_id"]] for row in unit.rows]
    scored = sum(score is not None for score in scores)
    if not scored:
        # a reference of 0 would pass every alpha
        raise _refuse("no benign screen prompt has a drifted chrF; the utility bar has nothing to stand on")
    # the same rule as for each alpha: every prompt counts, an unanswered one as 0
    mean = fmean(score or 0.0 for score in scores)
    ctx.run.note(f"drifted_chrf_subset = {mean:.4f} ({scored} of {len(scores)} scored, {ids.unparse(head)})")
    return mean


def _bytes(hub: Hub, source: RepoRef) -> int | None:
    try:
        files = hub.model_info(source, files_metadata=True).siblings or []
    except HfHubHTTPError as error:
        raise _refuse(f"{ids.unparse(source)} cannot be read ({type(error).__name__})") from error
    sizes = [file.size for file in files]
    return sum(sizes) if sizes and None not in sizes else None


def check_disk(directory: Path, sizes: Sequence[int | None], note: Callable[[str], object]) -> None:
    """Refuse downloads that do not fit the disk under ``directory`` with a tenth of their size to spare."""
    free = shutil.disk_usage(directory).free
    if None in sizes:
        note(f"a checkpoint size is unknown; {free / GIB:.1f} GiB of disk free")
        return
    wanted = sum(sizes) + sum(sizes) // 10
    if free < wanted:
        raise _refuse(f"the checkpoints need {wanted / GIB:.1f} GiB of disk, {free / GIB:.1f} GiB are free")


def stored_rows(ctx: Context, inputs: Inputs, alphas: Sequence[float]) -> dict[float, list[GenerationRow]]:
    """Per alpha the screen rows already uploaded, when they are of these weights, prompts and decoding."""
    head = RepoRef(inputs.responses, ctx.hub.dataset_info(inputs.responses).sha)
    scratch = tempfile.mkdtemp(prefix="driftmap-screen-")
    pattern = _rows_file(ctx, inputs.slug, "*")
    directory = Path(ctx.hub.snapshot_download(head, scratch, repo_type="dataset", allow_patterns=[pattern]))
    expected = sorted((unit.set, row["prompt_id"]) for unit in inputs.units for row in unit.rows)
    kept = {}
    for alpha in alphas:
        rows = read_rows(directory / _rows_file(ctx, inputs.slug, alpha), GenerationRow)
        # the name ties the rows to both revisions; a batch the backend failed on is generated again
        wanted = (inputs.name(alpha), inputs.prefix, inputs.decoding)
        same = all(
            (row.checkpoint, row.rendered_prefix_sha256, row.decoding) == wanted and row.stop_reason != "error"
            for row in rows
        )
        if same and sorted((row.set, row.prompt_id) for row in rows) == expected:
            kept[alpha] = rows
    return kept


def _row(alpha: float, rows: Sequence[GenerationRow], units: Sequence[Any]) -> AlphaRow:
    """The screen row of one alpha's generations, before any judge."""
    utility = screen_gen.utility(rows, units)
    failed = sum(row.stop_reason == "error" for row in rows)
    return AlphaRow(
        alpha=alpha,
        safe_rate_pooled=None,
        n_judged=0,
        chrf_pooled=utility.chrf,
        pass_safety=None,
        pass_chrf=False,
        # short of a batch, the rates would pool other prompts than the references
        error=f"{failed} screen rows failed in the backend" if failed else None,
        n_benign_scored=utility.scored,
        n_benign_empty=utility.empty,
        n_harmful=sum(row.set == "harmful" for row in rows),
        rows_sha256=hashlib.sha256(dump_rows(rows)).hexdigest(),
    )


def _error_row(alpha: float, reason: str) -> AlphaRow:
    return AlphaRow(
        alpha=alpha, safe_rate_pooled=None, n_judged=0, chrf_pooled=0.0, pass_safety=None, pass_chrf=False, error=reason
    )


def _of_rows(row: AlphaRow, rows: Sequence[GenerationRow], units: Sequence[Any]) -> bool:
    """True when the screen row was made from exactly these generations."""
    made = _row(row.alpha, rows, units)
    figures = ("rows_sha256", "n_harmful", "n_benign_scored", "n_benign_empty", "error")
    same = all(getattr(row, name) == getattr(made, name) for name in figures)
    return same and abs(row.chrf_pooled - made.chrf_pooled) <= select.TOLERANCE


def _dump(screen: AlphaScreen) -> bytes:
    return (screen.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def _put_screen(ctx: Context, screen: AlphaScreen, *, queued: bool = False) -> None:
    """Send the screen to the results repo, now or with the closing save."""
    send = ctx.flusher.add if queued else ctx.flusher.upload
    send(ctx.run.results, _screen_file(ctx), _dump(screen))


def _stored_screen(ctx: Context) -> AlphaScreen | None:
    """The screen in the results repo; None when there is none or it does not parse."""
    head = ctx.hub.dataset_info(ctx.run.results).sha
    path = _screen_file(ctx)
    scratch = tempfile.mkdtemp(prefix="driftmap-screen-")
    try:
        found = ctx.hub.hf_hub_download(f"{ctx.run.results}@{head}#{path}", scratch, repo_type="dataset")
        return AlphaScreen.model_validate_json(Path(found).read_bytes())
    except (RemoteEntryNotFoundError, ValidationError):
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def prepare(ctx: Context, device: str) -> Inputs:
    """Refuse what can be refused before any weights are downloaded; read the prompts and the chrF reference."""
    settings, hub, record, args = ctx.settings, ctx.hub, ctx.run.record, ctx.args
    mode = record.mode
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise _refuse("no CUDA device; this job does not run on CPU")
    if args.limit and mode == "full":
        raise _refuse(f"--limit is for smoke runs; a full screen takes {screen_gen.PER_LANGUAGE} prompts per language")
    ctx.flusher.resume_from = STAGE
    check_repos(settings)
    aligned = RepoRef(settings.sources.model, settings.revisions.get(settings.sources.model))
    if aligned.revision is None:
        raise _refuse(f"no pinned revision for {aligned.repo}; pin it and push revisions.json before launching")
    preamble = args.preamble
    if ids.preamble_note(preamble):
        ctx.run.note(ids.preamble_note(preamble))
    _, merged_repo = repo_names(settings, mode, ids.FINE_TUNES[args.fine_tune])
    if hub.model_info(merged_repo).private is not True:
        raise _refuse(f"{merged_repo} is not private; drifted weights are never read from an open repo")
    responses = f"{settings.owner}/{settings.repos.responses}"
    # private when new; one published since keeps its visibility
    hub.create_repo(responses, repo_type="dataset", private=True)

    work = Path(tempfile.mkdtemp(prefix="driftmap-repair-"))
    drifted, merged_sha = drifted_checkpoint(past_runs(hub, ctx.run.results, work / "records"), merged_repo, mode)
    eval_repo = f"{settings.owner}/{settings.repos.eval}"
    eval_ref = RepoRef(eval_repo, settings.revisions.get(eval_repo) or hub.dataset_info(eval_repo).sha)
    record.inputs += [
        Ref(kind="model", repo=drifted.repo, revision=drifted.revision, sha256=merged_sha),
        Ref(kind="model", repo=aligned.repo, revision=aligned.revision),
        Ref(kind="dataset", repo=eval_repo, revision=eval_ref.revision),
    ]
    record.merged_sha256 = merged_sha
    ctx.run.note(f"drifted = {ids.unparse(drifted)}")
    ctx.run.note(f"aligned = {ids.unparse(aligned)}")

    # small files first: these checks need no weights
    data_dir = Path(hub.snapshot_download(eval_ref, work / "data", repo_type="dataset", allow_patterns=PROMPT_FILES))
    tokenizer_dir = hub.snapshot_download(aligned, work / "tokenizer", allow_patterns=TOKENIZER_FILES)
    tokenizer = load_tokenizer(tokenizer_dir, CANARIES)
    if not (data_dir / "manifest.json").is_file():
        raise _refuse(f"manifest.json is missing from {eval_repo}")
    # from here the tokenizer renders under the run's condition
    prefix = check_rendering(tokenizer, json.loads((data_dir / "manifest.json").read_bytes()), preamble)
    record.bos_check = True
    per_language = min(screen_gen.PER_LANGUAGE, args.limit or screen_gen.PER_LANGUAGE)
    units = [
        unit
        for unit in read_units(data_dir, settings.languages, ids.slug(drifted), mode, per_language, preamble)
        if unit.set in screen_gen.SETS
    ]
    return Inputs(
        drifted=drifted,
        merged_sha=merged_sha,
        aligned=aligned,
        tokenizer=tokenizer,
        units=units,
        prefix=prefix,
        decoding=Decoding(runtime="transformers", max_new_tokens=settings.decoding.max_new_tokens),
        drifted_chrf=drifted_chrf(ctx, drifted, units, work),
        responses=responses,
        work=work,
    )


def _weights(ctx: Context, inputs: Inputs, copies: int) -> tuple[Path, Path, Any]:
    """Verified drifted directory, aligned directory, aligned model on CPU; room for ``copies`` drifted-sized ones."""
    sizes = [_bytes(ctx.hub, source) for source in (inputs.drifted, inputs.aligned)]
    check_disk(inputs.work, [*sizes, *[sizes[0]] * (copies - 1)], ctx.run.note)
    ctx.checkpoint()
    drifted_dir = fetch_verified(
        ctx.hub, inputs.drifted.repo, inputs.drifted.revision, inputs.merged_sha, inputs.work / "drifted"
    )
    aligned_dir = Path(ctx.hub.snapshot_download(inputs.aligned, inputs.work / "aligned"))
    ctx.checkpoint()
    return drifted_dir, aligned_dir, wiseft.load(aligned_dir, "cpu")


def _screen_alpha(
    ctx: Context, inputs: Inputs, drifted_dir: Path, aligned: Any, alpha: float, device: str
) -> list[GenerationRow]:
    """Load the drifted weights, interpolate, generate every screen prompt; the weights are dropped on return."""
    clock, batch = ctx.run.clock, ctx.settings.batch.screen
    name = inputs.name(alpha)
    started = clock.monotonic()
    model = wiseft.interpolated(drifted_dir, aligned, alpha, device, ctx.run.note)
    applied = clock.monotonic()
    generator = screen_gen.backend(model, inputs.tokenizer, name, batch)
    try:
        rows = screen_gen.screen(
            generator,
            inputs.tokenizer,
            inputs.units,
            checkpoint=name,
            decoding=inputs.decoding,
            batch_size=batch,
            between=ctx.checkpoint,
            system=render.system_message(ctx.args.preamble),
        )
    finally:
        generator.unload()
    minutes = [(applied - started) / 60, (clock.monotonic() - applied) / 60]
    ctx.run.note(f"alpha {alpha:g}: load and apply {minutes[0]:.1f} min, screen {minutes[1]:.1f} min")
    return rows


def run_screen(ctx: Context, inputs: Inputs, alphas: Sequence[float], device: str) -> None:
    """Generate and store every alpha's screen rows with their chrF; the safety verdict is made outside the job."""
    record, units = ctx.run.record, inputs.units
    ctx.run.note(f"alphas = {', '.join(f'{alpha:g}' for alpha in alphas)}")
    stored = _stored_screen(ctx)
    earlier = stored.alphas if stored else []
    found = stored_rows(ctx, inputs, sorted({*alphas, *(row.alpha for row in earlier)}))
    # rows of the stored screen made from generations of these weights keep their verdicts
    kept = {row.alpha: row for row in earlier if row.alpha in found and _of_rows(row, found[row.alpha], units)}
    rows = {alpha: found[alpha] for alpha in alphas if alpha in found} if ctx.args.resume else {}
    # a named alpha keeps its row until new rows are uploaded, so no flush shows part of a judged screen
    results = dict(kept)
    judge, baseline = (stored.judge, stored.baseline_safe_pooled) if kept else (None, None)
    drifted_dir, _, aligned = _weights(ctx, inputs, copies=1)
    owner = wiseft.tied_weight_owner(aligned)

    def screen_now() -> AlphaScreen:
        return select.choose_alpha(
            results.values(),
            baseline,
            inputs.drifted_chrf,
            ctx.settings.repair.bars,
            judge=judge,
            tied_weight_owner=owner,
            rule=select.rule_of(ctx.args.fine_tune),
        )

    for alpha in alphas:
        key = _rows_file(ctx, inputs.slug, alpha)
        if alpha in rows:
            ctx.run.note(f"alpha {alpha:g}: stored screen rows reused")
            ctx.flusher.mark_done(key)
            results[alpha] = kept.get(alpha) or _row(alpha, rows[alpha], units)
        else:
            ctx.checkpoint()
            try:
                rows[alpha] = _screen_alpha(ctx, inputs, drifted_dir, aligned, alpha, device)
            except Exception as error:
                failed = isinstance(error, wiseft.InterpolationFailed)
                reason = str(error) if failed else oom_text(error)
                if reason is None:
                    raise
                # the stored rows of a kept row are still what its verdict was made from
                results.setdefault(alpha, _error_row(alpha, reason))
                # queued, so the error itself is what ends the job
                _put_screen(ctx, screen_now(), queued=True)
                raise
            finally:
                _release()
            ctx.flusher.upload(inputs.responses, key, dump_rows(rows[alpha]), key=key)
            record.prompts += len(rows[alpha])
            record.prompt_tokens += sum(row.prompt_tokens for row in rows[alpha])
            record.generated_tokens += sum(row.generated_tokens for row in rows[alpha])
            # the same generations again keep their verdict
            unchanged = alpha in kept and _of_rows(kept[alpha], rows[alpha], units)
            results[alpha] = kept[alpha] if unchanged else _row(alpha, rows[alpha], units)
        _put_screen(ctx, screen_now())
        # a resume reuses an alpha by its stored rows and the stored screen, so the marker may lag behind them
        ctx.flusher.progress()
    ctx.run.note(f"recover_status = {screen_now().recover_status}")


def run_apply(ctx: Context, inputs: Inputs, alpha: float, device: str) -> None:
    """Push the repaired model at the alpha the decided screen names, with its provenance; refuses any other."""
    settings, mode = ctx.settings, ctx.run.record.mode
    stored = _stored_screen(ctx)
    named = stored is not None and stored.recover_status == "chosen" and stored.chosen is not None
    if not named or abs(stored.chosen - alpha) > select.TOLERANCE:
        says = f"it says {stored.recover_status}, chosen {stored.chosen}" if stored else "there is none"
        raise _refuse(f"the screen in {ctx.run.results} does not name alpha {alpha:g} as chosen ({says})")
    scored = [row for row in stored.alphas if row.error is None]
    found = stored_rows(ctx, inputs, [row.alpha for row in scored])
    loose = [row.alpha for row in scored if row.alpha not in found or not _of_rows(row, found[row.alpha], inputs.units)]
    if loose:
        which = ", ".join(f"{value:g}" for value in loose)
        made = f"the decided screen was not made from the stored rows of alpha {which} for {inputs.name(alpha)}"
        raise _refuse(f"{made}; screen and decide again")
    # the decision is read, never taken on trust: it must follow from its rows under the bars in force
    screen = select.choose_alpha(
        stored.alphas,
        stored.baseline_safe_pooled,
        inputs.drifted_chrf,
        settings.repair.bars,
        judge=stored.judge,
        tied_weight_owner=stored.tied_weight_owner,
        rule=select.rule_of(ctx.args.fine_tune),
    )
    if screen.recover_status != "chosen" or screen.chosen != stored.chosen:
        derived = f"{screen.recover_status}, chosen {screen.chosen}"
        raise _refuse(f"the stored decision for alpha {alpha:g} does not follow from its rows ({derived})")
    ctx.run.note(f"applied alpha = {alpha:g}, as the decided screen names it")
    ctx.run.note(f"chosen = {next(row for row in screen.alphas if row.alpha == screen.chosen).model_dump_json()}")
    repaired = repaired_repo(settings, mode, ids.FINE_TUNES[ctx.args.fine_tune])
    ctx.hub.create_repo(repaired, private=True)
    # creating over an existing repo leaves its visibility as it was
    if ctx.hub.model_info(repaired).private is not True:
        raise _refuse(f"{repaired} is not private; weights reach an open repo through the publish step only")
    drifted_dir, aligned_dir, aligned = _weights(ctx, inputs, copies=2)

    model = wiseft.interpolated(drifted_dir, aligned, alpha, device, ctx.run.note)
    del aligned
    _release()
    out = wiseft.save_repaired(
        model,
        inputs.work / "repaired",
        alpha,
        drifted=ids.unparse(inputs.drifted),
        aligned=ids.unparse(inputs.aligned),
        tokenizer_dir=aligned_dir,
    )
    (out / SCREEN_FILE).write_bytes(_dump(screen))
    commit = ctx.hub.upload_folder(repaired, out, commit_message=f"add the repaired model, alpha {alpha:g}")
    ctx.flusher.pushed(repaired, commit, sha256=merged_sha256(out))
    _put_screen(ctx, screen)


def asked_alphas(args: argparse.Namespace, settings: Settings, mode: str) -> list[float]:
    """The alphas a run works on: the one of ``--apply``, else those of ``--alphas``, else the configured ones.

    Raises DriftmapError (exit 12) on values the job cannot take; needs no Hub, so a launch can ask first.
    """
    if args.apply is not None and args.alphas is not None:
        raise _refuse("--apply pushes the chosen alpha and --alphas screens; give one of them")
    if args.apply is not None:
        alphas = parse_alphas(args.apply, "--apply")
        if len(alphas) != 1:
            raise _refuse(f"--apply takes one alpha, got {args.apply!r}")
        return alphas
    if args.alphas is not None:
        return parse_alphas(args.alphas, "--alphas")
    configured = [settings.repair.alpha] if mode == "smoke" else settings.repair.alphas
    return parse_alphas(",".join(map(repr, configured)), "[repair] alphas")


def body(ctx: Context, *, device: str = "cuda") -> None:
    """Screen the alphas, or with ``--apply`` push the model at the chosen one."""
    args, settings, mode = ctx.args, ctx.settings, ctx.run.record.mode
    args.preamble = condition(args)
    alphas = asked_alphas(args, settings, mode)
    inputs = prepare(ctx, device)
    if args.apply is not None:
        run_apply(ctx, inputs, alphas[0], device)
    else:
        run_screen(ctx, inputs, alphas, device)


def main(
    argv: Sequence[str] | None = None, *, client: Any = None, clock: Clock | None = None, device: str = "cuda"
) -> int:
    """Run the repair job; returns the exit code. ``device="cpu"`` is for runs without a GPU, such as tests."""
    job = functools.partial(body, device=device)
    return run_job(job, argv, kind="repair", arguments=_arguments, client=client, clock=clock)


if __name__ == "__main__":
    sys.exit(main())
