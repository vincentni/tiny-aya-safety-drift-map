"""Quant job: the repaired model as a q4_k_m GGUF, made with the pinned llama.cpp release on CPU."""

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
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context
from driftmap.quant import GGUF_FILE, GIB, QUANT, RECORD_FILE, convert, release, repo_names, require_disk
from driftmap.quant import record as gguf_record
from driftmap.scripts import run_job
from driftmap.scripts.drift import TOKENIZER_FILES

ASSETS = ("source_tarball", "cpu_asset")
WEIGHTS = ".safetensors"


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint", type=ids.parse, metavar="REPO@SHA", help="revision to convert; default: the head"
    )


def _source(ctx: Context, repo: str) -> RepoRef:
    """The repaired checkpoint to convert, pinned to a revision."""
    given = ctx.args.checkpoint
    if given is not None:
        if given.repo != repo or given.revision is None or given.file or given.steer:
            raise _refuse(f"--checkpoint must be {repo}@<sha> in {ctx.run.record.mode} mode")
        return given
    try:
        return RepoRef(repo, ctx.hub.model_info(repo).sha)
    except HfHubHTTPError as error:
        raise _refuse(f"{repo} cannot be read ({type(error).__name__}); run the repair job first") from error


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
    repaired, target = repo_names(settings, record.mode)
    source = _source(ctx, repaired)
    record.inputs.append(Ref(kind="model", repo=source.repo, revision=source.revision))
    weights = _weights_bytes(hub, source)
    work = Path(tempfile.mkdtemp(prefix="driftmap-quant-"))
    # the download, the bf16 file and the quantized one, plus the release
    require_disk(work, 3 * weights + GIB, "the conversion")
    config_dir = Path(hub.snapshot_download(source, work / "config", allow_patterns=TOKENIZER_FILES))
    architecture = _architecture(config_dir / "config.json", source)
    # private when new; one published since keeps its visibility
    hub.create_repo(target, private=True)

    roots = release.fetch_release(pinned, work / "release", opener, assets=ASSETS)
    for name in ASSETS:
        url, sha = getattr(pinned, name), getattr(pinned.sha256, name)
        record.inputs.append(Ref(kind="binary", repo=url, revision=pinned.tag, sha256=sha))
    converter = roots["source_tarball"] / convert.CONVERTER
    quantizer = release.binary(roots["cpu_asset"], convert.QUANTIZER)
    convert.check_converter(converter, architecture, run)

    ctx.checkpoint()
    model_dir = Path(hub.snapshot_download(source, work / "model"))
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
