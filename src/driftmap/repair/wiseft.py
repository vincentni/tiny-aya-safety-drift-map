"""WiSE-FT through SafeTune: interpolate the drifted weights toward the aligned model."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from driftmap.core.errors import OutOfMemory, classify_oom
from driftmap.core.redact import mask
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.gen.merged import TOKENIZER_FILES

# how SafeTune's own trainers name themselves in a provenance record
METHOD = "recover.WiseFTTrainer"
UNTIED = "untied"
EMBEDDINGS_ON_CPU = "embeddings interpolated on CPU after an out-of-memory apply"


class InterpolationFailed(DriftmapError):
    """The weights could not be interpolated in this machine's memory; no batch size changes that."""

    exit_code_default = ExitCode.COLUMN_FAILED


def load(directory: str | Path, device: str) -> Any:
    """The checkpoint in a local directory, on ``device`` in bf16 and eval mode; code shipped with it is refused."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(directory), dtype=torch.bfloat16, device_map=device, local_files_only=True, trust_remote_code=False
    )
    return model.eval()


def apply_alpha(model: Any, aligned: Any, alpha: float) -> Any:
    """Rewrite ``model`` as ``alpha * aligned + (1 - alpha) * model``, on its own device; raises OutOfMemory."""
    from safetune.runner.recover._whole_model import WiseFTTrainer

    known = aligned.state_dict().keys()
    missing = [name for name, _ in model.named_parameters() if name not in known]
    if missing:
        # the trainer would leave these as they drifted, without a word
        found = f"{len(missing)} parameters of the model are not in the aligned one, first {missing[0]}"
        raise DriftmapError(ExitCode.INTEGRITY, f"{found}; nothing was interpolated")
    try:
        return WiseFTTrainer(model, aligned_model=aligned, alpha=alpha).apply()
    except Exception as error:
        if classify_oom(error):
            raise OutOfMemory(mask(f"alpha {alpha:g}: {error}")[:200]) from error
        raise


def interpolated(
    directory: str | Path, aligned: Any, alpha: float, device: str, note: Callable[[str], object] = lambda text: None
) -> Any:
    """Load and apply; an out-of-memory apply is retried once with the embeddings on CPU; then InterpolationFailed."""
    from safetune.runner.utils.model_utils import free

    try:
        return apply_alpha(load(directory, device), aligned, alpha)
    except OutOfMemory:
        # the failed model may be half rewritten; it goes with the error once this block ends
        pass
    free()
    model = load(directory, device)
    embeddings = [module for module in (model.get_input_embeddings(), model.get_output_embeddings()) if module]
    # the arithmetic follows each tensor's device, so only the largest ones go through host memory
    for module in embeddings:
        module.to("cpu")
    try:
        apply_alpha(model, aligned, alpha)
    except OutOfMemory:
        model = None
    if model is None:
        # raised outside the handler: this is not a batch that a smaller batch size would fit
        free()
        tried = "on the device and again with the embeddings on CPU"
        raise InterpolationFailed(f"alpha {alpha:g}: the interpolation ran out of memory {tried}")
    for module in embeddings:
        module.to(device)
    note(f"alpha {alpha:g}: {EMBEDDINGS_ON_CPU}")
    return model


def tied_weight_owner(model: Any) -> str:
    """Name of the parameter the input and output embeddings share, ``UNTIED`` when they are separate."""
    shared = model.get_input_embeddings().weight
    output = model.get_output_embeddings()
    if output is None or output.weight is not shared:
        return UNTIED
    return next(name for name, weight in model.named_parameters() if weight is shared)


class _TokenizerFiles:
    """Stands in for a tokenizer at save time: its files are copied, never written again."""

    def __init__(self, directory: str | Path, name: str) -> None:
        self._directory = Path(directory)
        self.name_or_path = name

    def save_pretrained(self, target: str | Path) -> None:
        for pattern in TOKENIZER_FILES:
            for path in self._directory.glob(pattern):
                shutil.copy2(path, Path(target) / path.name)


def save_repaired(
    model: Any, out_dir: str | Path, alpha: float, *, drifted: str, aligned: str, tokenizer_dir: str | Path
) -> Path:
    """Save ``model`` into ``out_dir`` with a provenance record naming both inputs and the tokenizer files unchanged."""
    from safetune.provenance import input_entry
    from safetune.runner.utils.model_utils import save_checkpoint

    out_dir = Path(out_dir)
    # the record's source would otherwise be the directory the weights were loaded from
    model.config.name_or_path = drifted
    saved = save_checkpoint(
        model,
        _TokenizerFiles(tokenizer_dir, aligned),
        out_dir.name,
        out_dir=str(out_dir.parent),
        method=METHOD,
        inputs=[input_entry("model", aligned)],
        params={"alpha": alpha},
    )
    return Path(saved)
