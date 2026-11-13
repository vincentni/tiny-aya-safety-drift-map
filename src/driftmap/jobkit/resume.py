"""Pick up where an earlier run stopped."""

import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from huggingface_hub.errors import RemoteEntryNotFoundError
from pydantic import ValidationError

from driftmap.core.hub import Hub
from driftmap.core.records import PartialMarker

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "last-checkpoint"


def marker_path(run_id: str) -> str:
    """Where a run's partial.json lives in the results repo."""
    # outside runs/, so nothing reading records picks it up
    return f"partial/{run_id}.json"


def read_partial(
    hub: Hub, run_id: str, config_hash: str, *, results: str, note: Callable[[str], object] = logger.warning
) -> PartialMarker | None:
    """The marker of ``run_id``; None, with a note, when absent, torn or from another config; other failures raise."""
    head = hub.dataset_info(results).sha
    # a download needs a directory of its own
    scratch = tempfile.mkdtemp(prefix="driftmap-marker-")
    try:
        path = hub.hf_hub_download(f"{results}@{head}#{marker_path(run_id)}", scratch, repo_type="dataset")
        marker = PartialMarker.model_validate_json(Path(path).read_bytes())
    # an unreachable Hub is an error, not an absent marker
    except RemoteEntryNotFoundError:
        note(f"no marker for {run_id}; starting fresh")
        return None
    except ValidationError:
        note(f"marker of {run_id} is unreadable; starting fresh")
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if marker.config_hash != config_hash:
        note(f"marker of {run_id} was written under another config ({marker.config_hash[:12]}); starting fresh")
        return None
    return marker


def fetch_last_checkpoint(hub: Hub, adapter: str, workdir: str | Path) -> Path | None:
    """Download the trainer's last checkpoint into a new directory under ``workdir``; None when the repo has none."""
    head = hub.model_info(adapter).sha
    target = Path(tempfile.mkdtemp(prefix=f"{CHECKPOINT_DIR}-", dir=Path(workdir).resolve()))
    hub.snapshot_download(f"{adapter}@{head}", target, allow_patterns=f"{CHECKPOINT_DIR}/*")
    checkpoint = target / CHECKPOINT_DIR
    return checkpoint if checkpoint.is_dir() else None
