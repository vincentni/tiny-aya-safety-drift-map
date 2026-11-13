"""The merged drifted checkpoint: built once on CPU, verified by every job that loads it."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub.errors import RemoteEntryNotFoundError

from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef, merged_sha256
from driftmap.core.schemas import DriftmapError, ExitCode

SHARD = "model.safetensors"
INDEX = "model.safetensors.index.json"
TOKENIZER_FILES = ("tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*")


class MergedHashMismatch(DriftmapError):
    """The merged checkpoint is not the one the drift run recorded."""

    exit_code_default = ExitCode.INTEGRITY


def _load(directory: str | Path, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.bfloat16, local_files_only=True)
    return model.to(device)


def merge(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path) -> Path:
    """Fold the adapter into the model on CPU in bf16 and save it with the model's tokenizer files."""
    from peft import PeftModel

    out_dir = Path(out_dir)
    model = PeftModel.from_pretrained(_load(base_dir, "cpu"), str(adapter_dir)).merge_and_unload()
    model.save_pretrained(out_dir)
    # copied, not re-saved: a rewritten tokenizer could encode differently
    for pattern in TOKENIZER_FILES:
        for path in Path(base_dir).glob(pattern):
            shutil.copy2(path, out_dir / path.name)
    return out_dir


def _hub_sha256(hub: Hub, ref: RepoRef) -> str | None:
    """What ``merged_sha256`` returns for the shards as the Hub stores them; None without shards."""
    shards = hub.lfs_sha256(ref, [SHARD])
    if not shards:
        scratch = tempfile.mkdtemp(prefix="driftmap-index-")
        try:
            index = hub.hf_hub_download(RepoRef(ref.repo, ref.revision, INDEX), scratch)
            names = sorted(set(json.loads(Path(index).read_bytes())["weight_map"].values()))
        except RemoteEntryNotFoundError:
            return None
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        shards = hub.lfs_sha256(ref, names)
    return hashlib.sha256("".join(shards[name] for name in sorted(shards)).encode()).hexdigest()


def fetch_verified(hub: Hub, repo: str, revision: str, expected_sha: str, target: str | Path) -> Path:
    """Download into the empty ``target``; MergedHashMismatch unless the Hub and the disk hash to ``expected_sha``."""
    ref = RepoRef(repo, revision)
    name = f"{repo}@{revision[:12]}"
    on_hub = _hub_sha256(hub, ref)
    if on_hub != expected_sha:
        found = on_hub[:12] if on_hub else "no safetensors shards"
        raise MergedHashMismatch(f"{name}: the Hub holds {found}, the drift record says {expected_sha[:12]}")
    directory = Path(hub.snapshot_download(ref, target))
    local = merged_sha256(directory)
    if local != expected_sha:
        raise MergedHashMismatch(f"{name}: the download hashes to {local[:12]}, expected {expected_sha[:12]}")
    return directory


def verify_and_load(hub: Hub, repo: str, revision: str, expected_sha: str, device: str) -> Any:
    """The verified merged model on ``device`` in bf16; raises MergedHashMismatch."""
    directory = fetch_verified(hub, repo, revision, expected_sha, tempfile.mkdtemp(prefix="driftmap-merged-"))
    return _load(directory, device)
