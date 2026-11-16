"""Text-only baselines for the activation probes: what the prompt's words alone say of a label.

Two features, fitted on the same rows, split, pick rule and threshold rule as the activation probes: the length
of the prompt's context in tokens (one number, standardised) and a TF-IDF bag of the prompt's 1- and 2-grams.
A probe that does no better than these reads the text, not the model.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from driftmap.attribution import probe, probe_fit

# the TF-IDF features are not standardised, so their grid sits higher than the probes'
TEXT_C = (0.01, 0.1, 1.0, 10.0, 100.0)
LENGTH, TFIDF = FEATURES = ("length", "tfidf")
GRIDS = {LENGTH: probe.C_GRID, TFIDF: TEXT_C}
# lines that close a multiple-choice prompt after its context: the question, three options, the closing line
TAIL = 5


def model(feature: str, c: float):
    """The pipeline of one feature at one C, not yet fitted."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    regression = LogisticRegression(C=c, max_iter=probe_fit.MAX_ITER)
    if feature == LENGTH:
        return make_pipeline(StandardScaler(), regression)
    return make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True), regression)


def context_of(prompt: str, tail: int = TAIL) -> str:
    """The prompt without its last ``tail`` lines."""
    return "\n".join(prompt.split("\n")[:-tail])


def features(tokenizer, prompts: Sequence[str], tail: int = TAIL) -> dict[str, Any]:
    """Per feature what its pipeline reads of ``prompts``: the context's length in tokens of the model's
    tokenizer, without special tokens, and the prompt text itself."""
    lengths = [[len(tokenizer(context_of(prompt, tail), add_special_tokens=False)["input_ids"])] for prompt in prompts]
    return {LENGTH: np.array(lengths, dtype=float), TFIDF: list(prompts)}


def fit_one(feature: str, x_fit, y_fit, x_dev, y_dev) -> tuple[dict, dict]:
    """The picked pipeline of ``feature`` with its threshold and C, and the record of the pick; the rule is the
    activation probes'."""
    from sklearn.metrics import log_loss, roc_auc_score

    tried, models = [], {}
    for c in GRIDS[feature]:
        fitted = model(feature, c).fit(x_fit, y_fit)
        p_dev = fitted.predict_proba(x_dev)[:, 1]
        models[c] = (fitted, p_dev)
        tried.append(
            {
                "layer": 0,
                "C": c,
                "dev_auc": round(float(roc_auc_score(y_dev, p_dev)), 3),
                "dev_log_loss": round(float(log_loss(y_dev, p_dev)), 4),
                "fit_auc": round(float(roc_auc_score(y_fit, fitted.predict_proba(x_fit)[:, 1])), 3),
            }
        )
    best = probe_fit.choose(tried)
    fitted, p_dev = models[best["C"]]
    swept = probe_fit.sweep(y_dev, p_dev)
    chosen = probe_fit.threshold_of(swept)
    record = {
        "selected": {key: value for key, value in best.items() if key != "layer"},
        "threshold": chosen,
        "grid": [{key: value for key, value in row.items() if key != "layer"} for row in tried],
        "threshold_sweep": swept,
    }
    return {"pipeline": fitted, "threshold": chosen["threshold"], "C": best["C"]}, record


def fit(held: dict[str, Any], labels: np.ndarray, in_fit: np.ndarray) -> tuple[dict[str, dict], dict[str, dict]]:
    """Per feature the picked pipeline and its record, for one label; ``in_fit`` marks the rows of the fit split."""
    models, records = {}, {}
    for feature in FEATURES:
        x = held[feature]
        x_fit, x_dev = ([value for value, keep in zip(x, mask, strict=True) if keep] for mask in (in_fit, ~in_fit))
        if feature == LENGTH:
            x_fit, x_dev = np.array(x_fit), np.array(x_dev)
        models[feature], records[feature] = fit_one(feature, x_fit, labels[in_fit], x_dev, labels[~in_fit])
    return models, records
