"""Judge job: label one checkpoint's stored responses with PolyGuard, one (set, language) file at a time."""

import argparse
import functools
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError
from pydantic import ValidationError

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import GenerationRow, JudgeConfig, LabelRow, Ref, RunRecord
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context
from driftmap.jobkit.runrecord import peak_vram_gb, record_path
from driftmap.judge import parser, polyguard
from driftmap.judge.incremental import (
    Key,
    RawRow,
    check_unique,
    counts,
    dump_rows,
    judged_rows,
    key,
    pending,
    read_rows,
    union,
)
from driftmap.judge.polyguard import Judged, PolyGuardJudge
from driftmap.scripts import run_job

# benign rows are scored by chrF where they are generated, never judged
JUDGED_SETS = ("harmful", "xstest")
XSTEST_LANGUAGE = "en"
GIB = 2**30
# no output of this judge has been read before its first full run on a checkpoint
INSPECT_FIRST = 20
GENERATION_KINDS = ("generate", "steer", "quant_eval")

Unit = tuple[str, str]


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint", required=True, metavar="REPO@SHA", help="the checkpoint whose responses to judge"
    )
    parser.add_argument("--generation", required=True, metavar="RUN_ID", help="the run that generated them")
    parser.add_argument("--judge", choices=("qwen", "smol"), default="qwen", help="smol: smoke runs only")
    parser.add_argument(
        "--inspect-first", type=int, metavar="N", help="stop unless the first N outputs parse; 20 on a first full run"
    )
    parser.add_argument("--rejudge-unscored", action="store_true", help="judge rows labelled unscored again")


def units(settings: Settings) -> list[Unit]:
    """The (set, language) files a judge run labels, in order."""
    return [("harmful", language) for language in settings.languages] + [("xstest", XSTEST_LANGUAGE)]


def unit_key(slug: str, unit: Unit) -> str:
    return f"{slug}/{unit[0]}/{unit[1]}"


def _slug(checkpoint: str) -> str:
    try:
        return ids.slug(checkpoint)
    except ValueError:
        return checkpoint


def generation_rows(hub: Hub, results: RepoRef, responses: str, run_id: str, checkpoint: RepoRef, mode: str) -> RepoRef:
    """The responses repo at the revision generation run ``run_id`` recorded; refuses a run that is not ok."""
    scratch = tempfile.mkdtemp(prefix="driftmap-record-")
    try:
        path = hub.hf_hub_download(f"{ids.unparse(results)}#{record_path(run_id)}", scratch, repo_type="dataset")
        record = RunRecord.model_validate_json(Path(path).read_bytes())
    except RemoteEntryNotFoundError:
        raise _refuse(f"{results.repo} holds no record of run {run_id}") from None
    except ValidationError:
        raise _refuse(f"the record of run {run_id} is unreadable") from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if record.kind not in GENERATION_KINDS or record.status != "ok":
        raise _refuse(f"run {run_id} is a {record.kind} run that ended {record.status}; only a finished generation")
    if record.mode != mode:
        raise _refuse(f"run {run_id} is a {record.mode} run; a {mode} judge run reads {mode} responses")
    if not any((ref.repo, ref.revision) == (checkpoint.repo, checkpoint.revision) for ref in record.inputs):
        raise _refuse(f"run {run_id} did not read {ids.unparse(checkpoint)}")
    revision = next((ref.revision for ref in record.outputs if ref.kind == "dataset" and ref.repo == responses), None)
    if revision is None:
        raise _refuse(f"run {run_id} records no upload to {responses}")
    return RepoRef(responses, revision)


def read_inputs(
    directory: Path, slug: str, wanted: Sequence[Unit], mode: str, limit: int | None
) -> dict[Unit, list[GenerationRow]]:
    """Generation rows per unit; DriftmapError on a missing file, a repeated prompt or a row of another unit."""
    inputs = {}
    for name, language in wanted:
        path = ids.responses_path(slug, name, language, mode)
        rows = read_rows(directory / path, GenerationRow, path)
        if not rows:
            raise _refuse(f"{path} is missing or empty at the generation revision")
        check_unique(rows, path)
        stray = next(
            (row for row in rows if (_slug(row.checkpoint), row.set, row.language) != (slug, name, language)), None
        )
        if stray is not None:
            raise _refuse(f"{path} holds a row of {stray.checkpoint} {stray.set}/{stray.language}")
        inputs[name, language] = rows[: limit or None]
    return inputs


def _writable(hub: Hub, repo: str) -> RepoRef:
    """The dataset repo at its head; refuses one this job cannot write to."""
    try:
        hub.auth_check(repo, repo_type="dataset", write=True)
        return RepoRef(repo, hub.dataset_info(repo).sha)
    except HfHubHTTPError as error:
        raise _refuse(f"{repo} cannot be written by this job ({type(error).__name__})") from error


def _pinned_judge(hub: Hub, config: JudgeConfig) -> tuple[RepoRef, int | None]:
    """The judge at its pinned revision and the bytes its files take, None when the Hub does not say."""
    source = RepoRef(config.repo, config.revision)
    try:
        info = hub.model_info(source, files_metadata=True)
    except HfHubHTTPError as error:
        raise _refuse(f"the judge {ids.unparse(source)} cannot be read ({type(error).__name__})") from error
    sizes = [file.size for file in info.siblings or []]
    return source, sum(sizes) if sizes and None not in sizes else None


def check_disk(directory: Path, needed: int | None, note: Callable[[str], object]) -> None:
    """Refuse a download that does not fit the disk under ``directory`` with a tenth of its size to spare."""
    free = shutil.disk_usage(directory).free
    if needed is None:
        note(f"judge size unknown; {free / GIB:.1f} GiB of disk free")
        return
    wanted = needed + needed // 10
    if free < wanted:
        files = f"{needed / GIB:.1f} GiB of files plus a tenth"
        raise _refuse(f"the judge needs {wanted / GIB:.1f} GiB of disk ({files}), {free / GIB:.1f} GiB are free")


def load_judge(ctx: Context, source: RepoRef, config: JudgeConfig, work: Path, device: str, needed: int | None) -> Any:
    """Tokenizer first, checked; then the weights, once they are known to fit the disk."""
    hub, clock = ctx.hub, ctx.run.clock
    check_disk(work, needed, ctx.run.note)
    files = hub.snapshot_download(source, work / "tokenizer", allow_patterns=polyguard.TOKENIZER_FILES)
    tokenizer = polyguard.load_judge_tokenizer(files)
    started = clock.monotonic()
    weights = hub.snapshot_download(source, work / "model")
    ctx.run.note(f"judge download_s = {clock.monotonic() - started:.1f}")
    model = polyguard.load_model(weights, config.dtype, device)
    return PolyGuardJudge(model, tokenizer, config, ctx.settings.batch.judge)


def inspect(judged: Sequence[Judged]) -> None:
    """Exit 14 with the raw outputs printed unless every one of them parsed; unencodable pairs are left out."""
    judged = [answer for answer in judged if answer[0] != polyguard.UNENCODABLE]
    parsed = sum(labels is not None for _, labels in judged)
    if parsed == len(judged):
        return
    print(f"judge inspect: {parsed} of {len(judged)} outputs parse; the raw outputs follow", flush=True)
    for raw, _ in judged:
        print(json.dumps(raw, ensure_ascii=False), flush=True)
    raise DriftmapError(ExitCode.JUDGE_INSPECT_STOP, f"{parsed} of {len(judged)} judge outputs parse; fix the parser")


def _report(
    ctx: Context,
    labels: dict[str, list[LabelRow]],
    raws: dict[str, list[RawRow]],
    judge: PolyGuardJudge | None,
    sent: int,
) -> None:
    record = ctx.run.record
    record.prompts = sent
    if judge is not None:
        record.prompt_tokens, record.generated_tokens = judge.prompt_tokens, judge.generated_tokens
    peak = peak_vram_gb()
    if peak is not None:
        ctx.run.note(f"judge peak_vram_gb = {peak}")
    for language, rows in labels.items():
        ctx.run.note(f"labels {language} = {json.dumps(counts(rows), sort_keys=True)}")
        unencodable = sum(raw.raw == polyguard.UNENCODABLE for raw in raws[language])
        if unencodable:
            ctx.run.note(f"unencodable {language} = {unencodable}")


def body(ctx: Context, *, device: str = "cuda") -> None:
    """Refuse bad inputs before any weights download, then judge and upload unit by unit."""
    settings, hub, record, args = ctx.settings, ctx.hub, ctx.run.record, ctx.args
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise _refuse("no CUDA device; this job does not judge on CPU")
    smol = args.judge == "smol"
    if smol and record.mode != "smoke":
        raise _refuse(f"the small judge is for smoke runs; a full run takes {settings.judge.repo}")
    if settings.judge.parser_version != parser.PARSER_VERSION:
        configured = f"[judge] parser_version is {settings.judge.parser_version}"
        raise _refuse(
            f"{configured}, this code parses as {parser.PARSER_VERSION}; labels must name the parser they got"
        )
    config = polyguard.judge_config(settings, smol=smol)
    ctx.run.note(f"judge = {config.model_dump_json()}")
    source, judge_bytes = _pinned_judge(hub, config)
    try:
        checkpoint = ids.parse(args.checkpoint)
        slug = ids.slug(checkpoint)
    except ValueError as error:
        raise _refuse(f"--checkpoint: {error}") from None

    mode = record.mode
    # raw outputs go beside the responses, labels into the evaluation dataset
    responses = _writable(hub, f"{settings.owner}/{settings.repos.responses}")
    labelled = _writable(hub, f"{settings.owner}/{settings.repos.eval}")
    results = RepoRef(ctx.run.results, hub.dataset_info(ctx.run.results).sha)
    generated = generation_rows(hub, results, responses.repo, args.generation, checkpoint, mode)
    record.inputs += [
        Ref(kind="dataset", repo=generated.repo, revision=generated.revision),
        Ref(kind="model", repo=source.repo, revision=source.revision),
    ]
    work = Path(tempfile.mkdtemp(prefix="driftmap-judge-"))
    wanted = [ids.responses_path(slug, name, "*", mode) for name in JUDGED_SETS]
    rows_dir = hub.snapshot_download(generated, work / "rows", repo_type="dataset", allow_patterns=wanted)
    inputs = read_inputs(Path(rows_dir), slug, units(settings), mode, args.limit)

    def earlier(repo: RepoRef, name: str, pattern: str) -> Path:
        return Path(hub.snapshot_download(repo, work / name, repo_type="dataset", allow_patterns=[pattern]))

    kept_labels = earlier(labelled, "labels", ids.labels_path(slug, "*", mode))
    kept_raws = earlier(responses, "raws", ids.judge_raw_path(slug, "*", mode))
    languages = dict.fromkeys(language for _, language in inputs)
    labels = {code: read_rows(kept_labels / ids.labels_path(slug, code, mode), LabelRow) for code in languages}
    raws = {code: read_rows(kept_raws / ids.judge_raw_path(slug, code, mode), RawRow) for code in languages}
    waiting = {
        # the files say what is left, not an earlier run's marker
        unit: pending(rows, labels[unit[1]], config.config_hash, args.rejudge_unscored, raws=raws[unit[1]])
        for unit, rows in inputs.items()
    }

    judge = None
    # a fully labelled column costs no download
    if any(row.response is not None for rows in waiting.values() for row in rows):
        judge = load_judge(ctx, source, config, work, device, judge_bytes)
    answers: dict[Key, Judged] = {}

    def ask(rows: Sequence[GenerationRow]) -> None:
        new = [row for row in rows if row.response is not None and key(row) not in answers]
        if new:
            answers.update(zip(map(key, new), judge.judge([(row.prompt, row.response) for row in new]), strict=True))

    try:
        answered = [row for rows in waiting.values() for row in rows if row.response is not None]
        known = any(
            row.judge and row.judge.config_hash == config.config_hash for rows in labels.values() for row in rows
        )
        # by default only the first full run of this judge on the checkpoint is inspected
        default = 0 if mode == "smoke" or known else INSPECT_FIRST
        inspected = default if args.inspect_first is None else args.inspect_first
        first = answered[: max(inspected, 0)]
        ask(first)
        # before the first upload, so a parser fault leaves no label file behind
        inspect([answers[key(row)] for row in first])
        for unit, rows in waiting.items():
            language = unit[1]
            ask(rows)
            judged = [answers[key(row)] for row in rows if row.response is not None]
            new_labels, new_raws = judged_rows(rows, judged, config)
            labels[language] = union(labels[language], new_labels)
            raws[language] = union(raws[language], new_raws)
            if new_raws:
                ctx.flusher.upload(responses.repo, ids.judge_raw_path(slug, language, mode), dump_rows(raws[language]))
            if new_labels:
                # the raw outputs are there before a label says so
                path = ids.labels_path(slug, language, mode)
                ctx.flusher.upload(labelled.repo, path, dump_rows(labels[language]), key=unit_key(slug, unit))
            else:
                ctx.flusher.mark_done(unit_key(slug, unit))
            ctx.checkpoint()
    finally:
        _report(ctx, labels, raws, judge, len(answers))


def main(
    argv: Sequence[str] | None = None, *, client: Any = None, clock: Clock | None = None, device: str = "cuda"
) -> int:
    """Run the judge job; returns the exit code. ``device="cpu"`` is for runs without a GPU, such as tests."""
    return run_job(
        functools.partial(body, device=device), argv, kind="judge", arguments=arguments, client=client, clock=clock
    )


if __name__ == "__main__":
    sys.exit(main())
