"""The text-only baselines: what they read of a prompt, and the fit, pick and threshold of each."""

import numpy as np

from driftmap.attribution import probe, probe_baselines, probe_fit

CLOSE = ["Who was it?", "A. one person", "B. another person", "C. unknown", "Answer with one letter."]


class Words:
    """A token per word."""

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def prompts(count):
    """Open prompts have a short context that says somebody, settled ones a longer one that says because."""
    held, labels = [], []
    for at in range(count):
        open_ = at % 2 == 0
        said = f"somebody left item{at % 7}" if open_ else f"it was plain because the first one left item{at % 7} there"
        held.append("\n".join([said, *CLOSE]))
        labels.append(int(open_))
    return held, np.array(labels)


def test_the_length_is_that_of_the_context_in_tokens_and_the_bag_reads_the_whole_prompt():
    held, _ = prompts(4)

    found = probe_baselines.features(Words(), held)

    assert probe_baselines.context_of(held[0]) == "somebody left item0"
    assert probe_baselines.context_of("a passage\nof two lines\na question\na closing line", tail=2) == (
        "a passage\nof two lines"
    )
    assert found[probe_baselines.LENGTH].tolist() == [[3.0], [10.0], [3.0], [10.0]]
    assert found[probe_baselines.TFIDF] == held
    assert probe_baselines.features(Words(), held, tail=2)[probe_baselines.LENGTH][0].tolist() == [12.0]


def test_each_baseline_is_fitted_on_the_fit_rows_and_picked_and_cut_by_the_probes_rule():
    held, labels = prompts(80)
    in_fit = (np.arange(80) // 2) % 4 != 0
    read = probe_baselines.features(Words(), held)

    models, records = probe_baselines.fit(read, labels, in_fit)

    assert list(models) == list(records) == ["length", "tfidf"]
    for feature, grid in (("length", probe.C_GRID), ("tfidf", probe_baselines.TEXT_C)):
        record, model = records[feature], models[feature]
        assert [row["C"] for row in record["grid"]] == list(grid)
        assert all(set(row) == {"C", "dev_auc", "dev_log_loss", "fit_auc"} for row in record["grid"])
        assert record["selected"] == {
            key: value
            for key, value in probe_fit.choose([row | {"layer": 0} for row in record["grid"]]).items()
            if key != "layer"
        }
        assert record["selected"]["dev_auc"] == 1.0 and record["threshold"] == probe_fit.threshold_of(
            record["threshold_sweep"]
        )
        assert (model["C"], model["threshold"]) == (record["selected"]["C"], record["threshold"]["threshold"])
        x = read[feature][:4]
        assert (model["pipeline"].predict_proba(x)[:, 1] >= model["threshold"]).tolist() == [True, False, True, False]
    # the bag holds words and pairs of words that two prompts share, in lower case
    vocabulary = models["tfidf"]["pipeline"].steps[0][1].vocabulary_
    assert {"somebody", "somebody left", "because"} <= set(vocabulary) and "Who" not in vocabulary
    # the dev rows took no part in the fit: a word of theirs alone is not in the bag
    held[0] = held[0].replace("somebody", "somebody zebra")
    held[8] = held[8].replace("somebody", "somebody zebra")
    again, _ = probe_baselines.fit(probe_baselines.features(Words(), held), labels, in_fit)
    assert "zebra" not in again["tfidf"]["pipeline"].steps[0][1].vocabulary_
