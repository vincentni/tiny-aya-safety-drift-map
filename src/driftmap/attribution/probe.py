"""An activation probe that makes the base model abstain: what is frozen, and the files the probe steps share.

The probe reads the base model's hidden state at the last token of a multiple-choice prompt and says how likely it
is that the text does not settle the question. Where it fires, the answer of the gated system is the unknown
option; elsewhere the base model's own answer stands. The gate is computed after the fact, from answers that are
already stored and the probe's reading of the same prompt: no hook runs inside a serving model, and the base
model's weights are never changed.

The frozen probe is a standardisation and a logistic regression over one layer. It is kept as plain arrays in
``probe.safetensors`` with its layer, regularisation and threshold in ``probe.json``, so reading it runs no code
of the file's own; ``Probe.probability`` computes in float32, as the fitted pipeline does.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.special import expit

from driftmap.attribution import build

SEED = build.SEED
# hidden_states indices: 0 is the embedding output, the last one the output of the last block
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36)
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
WEIGHTS, SETTINGS = "probe.safetensors", "probe.json"
SCHEMA_WEIGHTS = "driftmap.probe-weights/1"
ARRAYS = ("mean", "scale", "coefficients", "intercept")
# the systems compared, and the label of a training row
BASE, GATED, TUNED = SYSTEMS = ("base", "base_plus_probe", "fine_tuned")
AMBIGUOUS = "ambiguous"


class Probe(NamedTuple):
    """The frozen probe: the layer it reads, its regularisation, the threshold it fires at, and its arrays."""

    layer: int
    c: float
    threshold: float
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    def probability(self, states: np.ndarray) -> np.ndarray:
        """Per row of ``states``, hidden states of the probe's layer: how likely the text leaves the answer open."""
        standard = (states.astype(np.float32) - self.mean) / self.scale
        # as a product of two matrices, the way the fitted pipeline computes it
        return expit((standard @ self.coefficients.reshape(-1, 1) + self.intercept).ravel())

    def fires(self, states: np.ndarray) -> np.ndarray:
        return self.probability(states) >= self.threshold


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def save(probe: Probe, folder: Path, **said: Any) -> None:
    """Write the probe under ``folder``: the arrays as float32 in ``probe.safetensors``; the layer, C, threshold,
    the hash of that file and what else is ``said`` of the probe in ``probe.json``."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.ascontiguousarray(getattr(probe, name), dtype=np.float32) for name in ARRAYS}
    save_file(arrays, folder / WEIGHTS)
    settings = {
        "schema": SCHEMA_WEIGHTS,
        "layer": probe.layer,
        "C": probe.c,
        "threshold": probe.threshold,
        "hidden_size": int(arrays["mean"].shape[0]),
        "reads": "the hidden state at the last token of the rendered prompt",
        "probability": "expit(((state - mean) / scale) @ coefficients + intercept), in float32",
        "weights_sha256": hashlib.sha256((folder / WEIGHTS).read_bytes()).hexdigest(),
    }
    (folder / SETTINGS).write_text(json.dumps(settings | said, indent=2) + "\n", encoding="utf-8")


def load(folder: Path) -> Probe:
    """The probe stored under ``folder``; stops when the arrays are not the ones its settings were written with."""
    settings = json.loads((folder / SETTINGS).read_text(encoding="utf-8"))
    if settings.get("schema") != SCHEMA_WEIGHTS:
        raise SystemExit(f"{SETTINGS} is not a probe's settings")
    if hashlib.sha256((folder / WEIGHTS).read_bytes()).hexdigest() != settings["weights_sha256"]:
        raise SystemExit(f"{WEIGHTS} is not the file {SETTINGS} was written with")
    arrays = load_file(folder / WEIGHTS)
    if sorted(arrays) != sorted(ARRAYS) or {arrays[name].shape for name in ARRAYS[:3]} != {(settings["hidden_size"],)}:
        raise SystemExit(f"{WEIGHTS} does not hold the arrays of a probe over {settings['hidden_size']} units")
    held = (arrays[name] for name in ARRAYS)
    return Probe(int(settings["layer"]), float(settings["C"]), float(settings["threshold"]), *held)


def load_states(folder: Path) -> tuple[dict[str, np.ndarray], list[int]]:
    """Hidden states by row id (layers x hidden size each) and the layers they hold, from every shard of
    ``folder``; stops when the shards hold different layers or none is there."""
    states, layers = {}, None
    for path in sorted(folder.glob("[0-9]*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            held = shard["layers"].tolist()
            if layers not in (None, held):
                raise SystemExit(f"{path.name} holds other layers than the shards before it")
            layers = held
            states.update(zip(shard["ids"].tolist(), shard["states"], strict=True))
    if layers is None:
        raise SystemExit(f"{folder.name} holds no shard of hidden states")
    return states, layers


def _kept(row: Mapping[str, Any], keep: Sequence[Mapping[str, Any]] | None) -> bool:
    return not keep or any(all(row.get(key) == value for key, value in wanted.items()) for wanted in keep)


def read_sets(spec: Path) -> dict[str, list[dict]]:
    """The evaluation sets a JSON file names, each as rows that carry the stored answers of the base model and of
    the fine-tune.

    The file holds ``{"sets": [...]}``; paths are taken from its folder. A set is one of:

    - ``{"name", "base", "fine_tuned", "keep"?}``: ``base`` and ``fine_tuned`` are answer files whose rows carry
      the prompt row and its ``response`` under one ``id``, as ``local_generate`` writes them; with ``keep``, a
      list of ``{field: value}``, only the rows that match one of them are read (``null`` matches a row without
      the field);
    - the same with ``"prompts"``: a file of prompt rows under ``prompt_id``, whose answers are joined from the two
      answer files as the responses dataset holds them;
    - ``{"name", "of": [names]}``: the rows of sets named before it, in that order.

    A row: ``id``, ``prompt``, ``condition``, ``texts`` (the option text per letter), ``letter_unknown``,
    ``letter_correct``, ``pair`` (the two rows of a base item in one language), ``cluster`` (the base item) and
    ``responses`` per system. Stops on a set that is no such entry and on an answer that is missing.
    """
    root, sets = spec.parent, {}
    for entry in json.loads(spec.read_text(encoding="utf-8"))["sets"]:
        name = entry["name"]
        if "of" in entry:
            sets[name] = [row for part in entry["of"] for row in sets[part]]
            continue
        key = "prompt_id" if "prompts" in entry else "id"
        answers = {system: read_jsonl(root / entry[system]) for system in (BASE, TUNED)}
        asked = read_jsonl(root / entry["prompts"]) if "prompts" in entry else answers[BASE]
        said = {system: {row[key]: row["response"] for row in rows} for system, rows in answers.items()}
        rows = []
        for row in asked:
            if not _kept(row, entry.get("keep")):
                continue
            lacking = [system for system in said if row[key] not in said[system]]
            if lacking:
                raise SystemExit(f"{name}: {row[key]} has no stored answer of {lacking[0]}")
            item = row.get("base_item") or row[key]
            rows.append(
                {
                    "id": row[key],
                    "prompt": row["prompt"],
                    "condition": row["condition"],
                    "texts": {option["letter"]: option["text"] for option in row["options"]},
                    "letter_unknown": row["letter_unknown"],
                    "letter_correct": row["letter_correct"],
                    "pair": f"{row.get('language', '')}/{item}",
                    "cluster": item,
                    "responses": {system: said[system][row[key]] for system in said},
                }
            )
        sets[name] = rows
    return sets
