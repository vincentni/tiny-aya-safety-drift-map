"""Transformers backend: a local checkpoint and its decoding settings."""

from pathlib import Path

from auditkit.model import AutoModel, Model
from auditkit.runspec import RunConfig

from . import MAX_NEW_TOKENS


def load_hf_model(checkpoint_dir: str | Path, dtype: str, batch_size: int) -> Model:
    """Loaded model over a local checkpoint; raises when the checkpoint cannot be loaded."""
    # the run config's batch size never reaches the pipeline
    model = AutoModel.resolve(f"hf:{checkpoint_dir}", dtype=dtype, batch_size=batch_size)
    # an empty call loads the weights, so a bad checkpoint fails here
    model.generate([])
    return model


def run_config(batch_size: int) -> RunConfig:
    """Greedy decoding; a backend failure surfaces on the first attempt."""
    return RunConfig(temperature=0, max_tokens=MAX_NEW_TOKENS, batch_size=batch_size, max_retries=0)
