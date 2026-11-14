"""Quant job: the repaired model, or the merged fine-tuned one, as a q4_k_m GGUF.

Made on CPU with the pinned llama.cpp release.
"""

import argparse
import functools
import json
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub.errors import HfHubHTTPError

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import Ref
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.gen.merged import fetch_verified
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context
from driftmap.quant import GGUF_FILE, GIB, QUANT, RECORD_FILE, SOURCES, convert, names, release, require_disk
from driftmap.quant import record as gguf_record
from driftmap.scripts import run_job
from driftmap.scripts.drift import TOKENIZER_FILES
from driftmap.scripts.generate import past_runs

ASSETS = ("source_tarball", "cpu_asset")
WEIGHTS = ".safetensors"


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=ids.parse,
        metavar="REPO@SHA",
        help="revision to convert; default: the head, or for merged weights the latest fine-tune run's",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=SOURCES[0],
        help="checkpoint to convert: the repaired model, or the merged fine-tuned one",
    )


def _named(ctx: Context, repo: str) -> RepoRef | None:
    """The ``--checkpoint`` given, None without one; refuses anything but a revision of ``repo``."""
    given = ctx.args.checkpoint
    if given is not None and (given.repo != repo or given.revision is None or given.file or given.steer):
        raise _refuse(f"--checkpoint must be {repo}@<sha> in {ctx.run.record.mode} mode")
    return given


def _source(ctx: Context, repo: str) -> RepoRef:
    """The repaired checkpoint to convert, pinned to a revision."""
    given = _named(ctx, repo)
    if given is not None:
        return given
    try:
        return RepoRef(repo, ctx.hub.model_info(repo).sha)
    except HfHubHTTPError as error:
        raise _refuse(f"{repo} cannot be read ({type(error).__name__}); run the repair job first") from error


def _merged_source(ctx: Context, repo: str, records: Path) -> tuple[RepoRef, str]:
    """The merged fine-tuned checkpoint to convert and the weights hash its fine-tune run recorded.

    The revision is the one ``--checkpoint`` names, else the one the latest finished fine-tune run of this mode
    pushed; a revision no such run recorded is refused, as is a repo that is not private.
    """
    given, mode = _named(ctx, repo), ctx.run.record.mode
    try:
        private = ctx.hub.model_info(repo).private
    except HfHubHTTPError as error:
        raise _refuse(f"{repo} cannot be read ({type(error).__name__}); run the fine-tune job first") from error
    if private is not True:
        raise _refuse(f"{repo} is not private; merged weights are never read from an open repo")
    for run in reversed(past_runs(ctx.hub, ctx.run.results, records)):
        pushed = next((ref for ref in run.outputs if ref.repo == repo), None)
        if run.kind != "drift" or run.status != "ok" or not run.merged_sha256 or pushed is None:
            continue
        if pushed.revision == given.revision if given else run.mode == mode:
            return RepoRef(repo, pushed.revision), run.merged_sha256
    which = f"at {given.revision[:12]}" if given else f"in {mode} mode"
    raise _refuse(f"no finished drift record for {repo} {which}; the merged weights cannot be verified")


def _weights_bytes(hub: Hub, source: RepoRef) -> int:
    """Bytes of the checkpoint's safetensors files; refuses a revision without any."""
    try:
        files = hub.model_info(source, files_metadata=True).siblings or []
    except HfHubHTTPError as error:
        raise _refuse(f"{ids.unparse(source)} cannot be read ({type(error).__name__})") from error
    sizes = [file.size or 0 for file in files if file.rfilename.endswith(WEIGHTS)]
    if not sizes:
        raise _refuse(f"{ids.unparse(source)} holds no safetensors weights")
    return sum(sizes)


def _architecture(config: Path, source: RepoRef) -> str:
    try:
        return json.loads(config.read_bytes())["architectures"][0]
    except (OSError, ValueError, LookupError, TypeError):
        raise _refuse(f"config.json of {ids.unparse(source)} names no architecture") from None


def body(
    ctx: Context, *, opener: Callable[..., Any] = urllib.request.urlopen, run: Callable[..., Any] = subprocess.run
) -> None:
    """Verify the release and the converter before the weights download, then convert, quantize and upload."""
    settings, hub, record = ctx.settings, ctx.hub, ctx.run.record
    pinned = settings.llamacpp_release
    kept = [name for source in SOURCES for name in names(settings, record.mode, source)]
    if len(set(map(str.casefold, kept))) < len(kept):
        raise _refuse("two of the repos in the settings share a name; give each its own")
    checkpoint, target = names(settings, record.mode, ctx.args.source)
    work = Path(tempfile.mkdtemp(prefix="driftmap-quant-"))
    if ctx.args.source == "merged":
        source, merged_sha = _merged_source(ctx, checkpoint, work / "records")
    else:
        source, merged_sha = _source(ctx, checkpoint), None
    record.inputs.append(Ref(kind="model", repo=source.repo, revision=source.revision, sha256=merged_sha))
    weights = _weights_bytes(hub, source)
    # the download, the bf16 file and the quantized one, plus the release
    require_disk(work, 3 * weights + GIB, "the conversion")
    config_dir = Path(hub.snapshot_download(source, work / "config", allow_patterns=TOKENIZER_FILES))
    architecture = _architecture(config_dir / "config.json", source)
    # private when new; one published since keeps its visibility
    hub.create_repo(target, private=True)
    if merged_sha is not None and hub.model_info(target).private is not True:
        raise _refuse(f"{target} is not private; weights reach an open repo through the publish step only")

    roots = release.fetch_release(pinned, work / "release", opener, assets=ASSETS)
    for name in ASSETS:
        url, sha = getattr(pinned, name), getattr(pinned.sha256, name)
        record.inputs.append(Ref(kind="binary", repo=url, revision=pinned.tag, sha256=sha))
    converter = roots["source_tarball"] / convert.CONVERTER
    quantizer = release.binary(roots["cpu_asset"], convert.QUANTIZER)
    convert.check_converter(converter, architecture, run)

    ctx.checkpoint()
    if merged_sha is None:
        model_dir = Path(hub.snapshot_download(source, work / "model"))
    else:
        # only the weights the fine-tune run hashed are converted
        model_dir = fetch_verified(hub, source.repo, source.revision, merged_sha, work / "model")
        record.merged_sha256 = merged_sha
    ctx.checkpoint()
    bf16 = convert.convert(converter, model_dir, work / f"model-{convert.OUTTYPE}.gguf", run)
    ctx.checkpoint()
    out = work / "out"
    out.mkdir()
    gguf = convert.quantize(quantizer, bf16, out / GGUF_FILE, run)
    # reproducible from the pinned checkpoint and release, so it is not uploaded
    bf16.unlink()
    described = gguf_record.write(gguf, source, pinned, out / RECORD_FILE)
    ctx.checkpoint()

    clock = ctx.run.clock
    megabytes = sum(path.stat().st_size for path in out.iterdir()) / 2**20
    started = clock.monotonic()
    message = f"add the {QUANT} GGUF of {source.repo}@{source.revision[:12]}"
    commit = hub.upload_folder(target, out, commit_message=message)
    seconds = clock.monotonic() - started
    ctx.flusher.pushed(target, commit, sha256=described.sha256)
    if record.mode == "smoke" and seconds > 0:
        record.upload_mb_s = round(megabytes / seconds, 2)
    # the argument the quant-eval job takes
    ctx.run.note(f"gguf = {target}@{commit.oid}#{GGUF_FILE}")


def main(
    argv: Sequence[str] | None = None,
    *,
    client: Any = None,
    clock: Clock | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    run: Callable[..., Any] = subprocess.run,
) -> int:
    """Run the quant job; returns the exit code. Tests replace ``opener`` and ``run``."""
    job = functools.partial(body, opener=opener, run=run)
    return run_job(job, argv, kind="quant", arguments=_arguments, client=client, clock=clock)


if __name__ == "__main__":
    sys.exit(main())
