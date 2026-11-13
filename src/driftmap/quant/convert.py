"""Conversion steps: the release's converter to a bf16 GGUF, then llama-quantize to q4_k_m."""

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from driftmap.core.redact import mask
from driftmap.core.schemas import DriftmapError, ExitCode

from . import QUANT

logger = logging.getLogger(__name__)

CONVERTER = "convert_hf_to_gguf.py"
QUANTIZER = "llama-quantize"
OUTTYPE = "bf16"
LIST_MODELS = "--print-supported-models"
CHECK_TIMEOUT_S = 300
TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
TAIL = 300
RUNG = (
    "next rung: build llama.cpp at the same tag inside the job with cmake, "
    "then the llama.cpp full image over a bucket volume"
)

Run = Callable[..., Any]


class ConversionFailed(DriftmapError):
    """A conversion step failed or its program did not start; the quantized column fails."""

    exit_code_default = ExitCode.COLUMN_FAILED


def tool_env() -> dict[str, str]:
    """This process's environment without the Hub token and with the Hub switched off."""
    env = {name: value for name, value in os.environ.items() if name not in TOKEN_VARIABLES}
    return env | {"HF_HUB_OFFLINE": "1"}


def tail(text: object) -> str:
    """The end of a program's output as one masked line."""
    # masked whole: a cut could leave half a token unrecognised
    return mask(" ".join(str(text or "").split()))[-TAIL:]


def _output(result: Any) -> str:
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def _run(step: str, argv: list[str], out: Path, run: Run) -> Path:
    try:
        result = run(argv, check=True, capture_output=True, text=True, errors="replace", env=tool_env())
    except subprocess.CalledProcessError as error:
        raise ConversionFailed(f"{step} exited {error.returncode}: {tail(_output(error))}; {RUNG}") from error
    except OSError as error:
        raise ConversionFailed(f"{step} did not start ({type(error).__name__}: {error}); {RUNG}") from error
    logger.info("%s: %s", step, tail(_output(result)))
    if not out.is_file() or out.stat().st_size == 0:
        raise ConversionFailed(f"{step} wrote no file at {out.name}; {RUNG}")
    return out


def check_converter(converter: Path, architecture: str, run: Run = subprocess.run) -> None:
    """Raise DriftmapError (exit 12) unless the converter starts in this environment and lists ``architecture``."""
    if not converter.is_file():
        raise DriftmapError(ExitCode.INTEGRITY, f"the llama.cpp source holds no {converter.name}")
    argv = [sys.executable, str(converter), LIST_MODELS]
    try:
        result = run(
            argv, check=True, capture_output=True, text=True, errors="replace", env=tool_env(), timeout=CHECK_TIMEOUT_S
        )
    except (subprocess.SubprocessError, OSError) as error:
        detail = tail(_output(error)) if isinstance(error, subprocess.SubprocessError) else ""
        reason = f"{type(error).__name__}: {detail or error}"
        raise DriftmapError(ExitCode.INTEGRITY, f"the converter does not start here ({reason})") from error
    if architecture not in _output(result).split():
        raise DriftmapError(
            ExitCode.INTEGRITY, f"the converter does not list {architecture}; this release cannot convert the model"
        )


def convert(converter: Path, model_dir: Path, out: Path, run: Run = subprocess.run) -> Path:
    """Write the checkpoint in ``model_dir`` as a bf16 GGUF at ``out``; raises ConversionFailed (exit 13)."""
    argv = [sys.executable, str(converter), str(model_dir), "--outtype", OUTTYPE, "--outfile", str(out)]
    return _run("convert", argv, Path(out), run)


def quantize(quantizer: Path, bf16: Path, out: Path, run: Run = subprocess.run) -> Path:
    """Quantize the bf16 GGUF to q4_k_m at ``out``; raises ConversionFailed (exit 13)."""
    return _run("quantize", [str(quantizer), str(bf16), str(out), QUANT], Path(out), run)
