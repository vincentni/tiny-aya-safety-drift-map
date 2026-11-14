"""Stand-in for SafeTune 0.1.10 at its import paths: the WiSE-FT trainer and the checkpoint writer."""

import gc
import json
import os
import sys
import types
from typing import Any

import torch

TRAINER_MODULE = "safetune.runner.recover._whole_model"
SAVE_MODULE = "safetune.runner.utils.model_utils"
PROVENANCE_MODULE = "safetune.provenance"
PROVENANCE_FILE = "lexsi_provenance.json"
VERSION = "0.1.10"


def input_entry(kind: str, ref: Any, config: str | None = None) -> dict:
    entry = {"kind": kind, "ref": None if ref is None else str(ref)}
    if kind == "dataset":
        entry["config"] = config
    entry["provenance"] = None
    return entry


class FakeSafeTune:
    """Records every apply and save; ``raise_at`` maps an alpha to the error its apply raises, ``raise_once`` once."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.raise_at: dict[float, BaseException] = {}
        self.raise_once: dict[float, BaseException] = {}
        fake = self

        class WiseFTTrainer:
            PILLAR = "recover"
            METHOD = "WiseFTTrainer"

            def __init__(self, model=None, *, aligned_model=None, alpha: float = 0.5, **kwargs):
                self.model = model
                self.aligned_model = aligned_model
                self.alpha = alpha
                fake.calls.append(("init", {"alpha": alpha, **kwargs}))

            def apply(self, *, alpha: float | None = None, **kwargs):
                alpha = self.alpha if alpha is None else alpha
                fake.calls.append(("apply", alpha))
                error = fake.raise_once.pop(alpha, None) or fake.raise_at.get(alpha)
                if error is not None:
                    raise error
                if not 0.0 <= alpha <= 1.0:
                    raise ValueError(f"alpha must be in [0, 1], got {alpha}")
                aligned = self.aligned_model.state_dict()
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name not in aligned:
                            continue
                        toward = aligned[name].to(param.device, dtype=torch.float32)
                        param.copy_((alpha * toward + (1.0 - alpha) * param.float()).to(param.dtype))
                return self.model

        def save_checkpoint(
            model,
            tokenizer,
            name: str,
            *,
            out_dir: str | None = None,
            safe_serialization: bool = True,
            processor=None,
            method: str | None = None,
            inputs: list | None = None,
            params: dict | None = None,
        ) -> str:
            fake.calls.append(("save", {"method": method, "params": params}))
            path = os.path.join(out_dir, name)
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path, safe_serialization=safe_serialization)
            tokenizer.save_pretrained(path)
            source = input_entry("model", getattr(model.config, "_name_or_path", None) or None)
            record = {
                "schema": "lexsi.provenance/1",
                "library": "safetune",
                "version": VERSION,
                "git_sha": None,
                "created_at": "2026-11-13T00:00:00Z",
                "base_model": source["ref"],
                "method": method,
                "inputs": [source, *(inputs or [])],
                "params": dict(params or {}),
            }
            with open(os.path.join(path, PROVENANCE_FILE), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, indent=2) + "\n")
            return path

        def free():
            fake.calls.append(("free", None))
            gc.collect()

        self.WiseFTTrainer = WiseFTTrainer
        self.save_checkpoint = save_checkpoint
        self.free = free

    @property
    def applied(self) -> list[float]:
        return [alpha for name, alpha in self.calls if name == "apply"]

    def install(self, monkeypatch) -> "FakeSafeTune":
        """Put the fake where the job imports SafeTune from; the real package is never loaded."""
        members = {
            TRAINER_MODULE: {"WiseFTTrainer": self.WiseFTTrainer},
            SAVE_MODULE: {"save_checkpoint": self.save_checkpoint, "free": self.free},
            PROVENANCE_MODULE: {"input_entry": input_entry, "FILENAME": PROVENANCE_FILE},
        }
        for path, names in members.items():
            parts = path.split(".")
            # every parent package, so the import never reaches the installed one
            for depth in range(1, len(parts)):
                parent = ".".join(parts[:depth])
                if not isinstance(sys.modules.get(parent), _Stub):
                    monkeypatch.setitem(sys.modules, parent, _Stub(parent))
            module = _Stub(path)
            for name, value in names.items():
                setattr(module, name, value)
            monkeypatch.setitem(sys.modules, path, module)
        return self


class _Stub(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        # a package, so submodule imports resolve through sys.modules
        self.__path__: list[str] = []
