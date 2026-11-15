"""Fit the activation probe on the fit rows, choose it on the dev rows, and freeze it.

    python -I -m driftmap.attribution.probe_fit --rows OUT/probe_train_in.jsonl --states OUT/acts/train --out OUT

``--rows`` is the training file ``probe_rows`` wrote and ``--states`` the folder ``probe_extract`` wrote for it.
No evaluation set is read. Per layer and per C a standardisation and an L2 logistic regression (scikit-learn,
lbfgs) is fitted on the fit rows alone and scored on the dev rows. The rule that picks one is fixed here:

- the highest dev AUC, rounded to 3 decimals; ties by the lower dev log loss, rounded to 4 decimals, then the
  smaller C, then the lower layer;
- its threshold, of 0.05, 0.10, ..., 0.95: the one with the highest dev true-positive rate minus false-positive
  rate; ties by the value nearest 0.5, then the lower one.

The pick is kept as fitted on the fit rows; it is not fitted again on dev. Written under ``--out``: the probe
(``probe.safetensors`` and ``probe.json``, see ``probe``) and ``probe_fit.json``, the dev figures of every layer
and C, the pick and the sweep of thresholds.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from driftmap.attribution import probe

FIT = "probe_fit.json"
MAX_ITER = 5000


def choose(tried: Sequence[dict]) -> dict:
    """The entry of the grid the rule picks; each entry holds its layer, C, dev AUC and dev log loss, rounded."""
    return min(tried, key=lambda row: (-row["dev_auc"], row["dev_log_loss"], row["C"], row["layer"]))


def sweep(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Per threshold, the share of open rows and of settled rows the probe fires on."""
    rows = []
    for threshold in probe.THRESHOLDS:
        fired = scores >= threshold
        rates = (round(float(fired[labels == label].mean()), 4) for label in (1, 0))
        rows.append(
            dict(zip(("threshold", "true_positive_rate", "false_positive_rate"), (threshold, *rates), strict=True))
        )
    return rows


def threshold_of(swept: Sequence[dict]) -> dict:
    """The entry of the sweep the rule picks."""
    return min(
        swept,
        key=lambda row: (
            -(row["true_positive_rate"] - row["false_positive_rate"]),
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )


def fit(rows: Sequence[dict], states: dict[str, np.ndarray], layers: Sequence[int]) -> tuple[probe.Probe, dict]:
    """The frozen probe and the record of how it was chosen, from the training rows and their hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parts = {split: [row for row in rows if row["split"] == split] for split in ("fit", "dev")}
    y_fit, y_dev = (np.array([row["label"] for row in parts[split]]) for split in ("fit", "dev"))
    tried, models = [], {}
    for place, layer in enumerate(layers):
        x_fit, x_dev = (np.stack([states[row["id"]][place] for row in parts[split]]) for split in ("fit", "dev"))
        for c in probe.C_GRID:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITER))
            model.fit(x_fit, y_fit)
            p_dev = model.predict_proba(x_dev)[:, 1]
            models[layer, c] = (model, p_dev, x_dev)
            tried.append(
                {
                    "layer": layer,
                    "C": c,
                    "dev_auc": round(float(roc_auc_score(y_dev, p_dev)), 3),
                    "dev_log_loss": round(float(log_loss(y_dev, p_dev)), 4),
                    "fit_auc": round(float(roc_auc_score(y_fit, model.predict_proba(x_fit)[:, 1])), 3),
                }
            )
    best = choose(tried)
    model, p_dev, x_dev = models[best["layer"], best["C"]]
    swept = sweep(y_dev, p_dev)
    chosen = threshold_of(swept)
    scaler, regression = model.steps[0][1], model.steps[1][1]
    arrays = (scaler.mean_, scaler.scale_, regression.coef_.ravel(), regression.intercept_)
    frozen = probe.Probe(best["layer"], best["C"], chosen["threshold"], *(np.asarray(a, np.float32) for a in arrays))
    # the arrays alone must read the dev rows as the fitted pipeline does
    again = frozen.probability(x_dev)
    if not np.allclose(again, p_dev, atol=1e-6) or ((again >= frozen.threshold) != (p_dev >= frozen.threshold)).any():
        raise SystemExit("the stored arrays do not give the fitted pipeline's probabilities on the dev rows")
    fired = p_dev >= chosen["threshold"]
    record = {
        "fit_rows": len(parts["fit"]),
        "dev_rows": len(parts["dev"]),
        "selected": best,
        "threshold": chosen,
        "dev_ambiguous_rows": int(y_dev.sum()),
        "dev_disambiguated_rows": int((1 - y_dev).sum()),
        "dev_fired_on_ambiguous": int(fired[y_dev == 1].sum()),
        "dev_fired_on_disambiguated": int(fired[y_dev == 0].sum()),
        "threshold_sweep": swept,
        "grid": tried,
    }
    return frozen, record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=Path, required=True, help="the training file probe_rows wrote")
    parser.add_argument("--states", type=Path, required=True, help="the folder of hidden states of those rows")
    parser.add_argument("--out", type=Path, required=True, help="folder for the probe and the record of its choice")
    args = parser.parse_args(argv)
    rows = probe.read_jsonl(args.rows)
    states, layers = probe.load_states(args.states)
    if set(states) != {row["id"] for row in rows}:
        raise SystemExit(f"{len(states)} hidden states for {len(rows)} training rows")
    frozen, record = fit(rows, states, layers)
    grid = {"layers": list(layers), "C": list(probe.C_GRID), "thresholds": list(probe.THRESHOLDS)}
    probe.save(frozen, args.out, layers_extracted=list(layers), chosen_from=grid, fitted_on=args.rows.name)
    (args.out / FIT).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    kept = {key: value for key, value in record.items() if key not in ("grid", "threshold_sweep")}
    print(json.dumps(kept, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
