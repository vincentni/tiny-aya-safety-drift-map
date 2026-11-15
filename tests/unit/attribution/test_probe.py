"""The activation probe: its rows, the hidden states it reads, its fit and choice, its safe file and its scores."""

import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from driftmap.attribution import bbq, build, probe, probe_extract, probe_fit, probe_rows, probe_scores
from driftmap.core import ids
from driftmap.table.probe import ProbeTables
from tests.fakes import bbq as fake_bbq

FOLDER = ids.training_folder(ids.TRAIN_SETS[0])
OPTIONS = [{"letter": letter, "text": f"option text {letter}"} for letter in "ABC"]


def no_download(url):
    raise AssertionError(f"asked for {url}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def separable(count, hidden=6, seed=5):
    """Hidden states of two kinds of row that the third unit tells apart, as float32, and their labels."""
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    states = rng.normal(size=(count, hidden)).astype(np.float32)
    states[:, 2] += 4 * (2 * labels - 1)
    return states, labels


def test_the_probe_is_kept_as_plain_arrays_that_read_states_as_the_fitted_pipeline_does(tmp_path):
    states, labels = separable(200)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(states, labels)
    scaler, regression = pipeline.steps[0][1], pipeline.steps[1][1]
    arrays = (scaler.mean_, scaler.scale_, regression.coef_.ravel(), regression.intercept_)
    fitted = probe.Probe(20, 0.1, 0.35, *arrays)

    probe.save(fitted, tmp_path / "frozen", fitted_on="rows.jsonl")
    frozen = probe.load(tmp_path / "frozen")

    assert sorted(path.name for path in (tmp_path / "frozen").iterdir()) == ["probe.json", "probe.safetensors"]
    assert (frozen.layer, frozen.c, frozen.threshold) == (20, 0.1, 0.35)
    assert {getattr(frozen, name).dtype for name in probe.ARRAYS} == {np.dtype("float32")}
    wanted = pipeline.predict_proba(states)[:, 1]
    assert np.array_equal(frozen.probability(states), wanted)
    assert np.array_equal(frozen.fires(states), wanted >= 0.35) and 0 < frozen.fires(states).sum() < len(states)
    settings = json.loads((tmp_path / "frozen" / "probe.json").read_text(encoding="utf-8"))
    weights = (tmp_path / "frozen" / "probe.safetensors").read_bytes()
    assert settings["weights_sha256"] == hashlib.sha256(weights).hexdigest()
    assert (settings["hidden_size"], settings["fitted_on"], settings["C"]) == (6, "rows.jsonl", 0.1)
    assert "last token of the rendered prompt" in settings["reads"]

    # other arrays than the settings were written with, and settings of something else, are not read
    probe.save(fitted._replace(mean=fitted.mean + 1), tmp_path / "other")
    (tmp_path / "frozen" / "probe.safetensors").write_bytes((tmp_path / "other" / "probe.safetensors").read_bytes())
    with pytest.raises(SystemExit, match="probe.safetensors is not the file probe.json was written with"):
        probe.load(tmp_path / "frozen")
    (tmp_path / "other" / "probe.json").write_text(json.dumps(settings | {"schema": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="probe.json is not a probe's settings"):
        probe.load(tmp_path / "other")


def shard(folder, name, ids_, states, layers=(4, 8)):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", ids=np.array(ids_), states=states, layers=np.array(layers))


def test_hidden_states_are_read_by_row_id_from_every_shard(tmp_path):
    states = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    shard(tmp_path / "acts", 0, ["a", "c"], states[:2])
    shard(tmp_path / "acts", 1, ["b", "d"], states[2:])

    found, layers = probe.load_states(tmp_path / "acts")

    assert layers == [4, 8] and sorted(found) == ["a", "b", "c", "d"]
    assert np.array_equal(found["c"], states[1]) and np.array_equal(found["b"], states[2])
    shard(tmp_path / "acts", 2, ["e"], states[:1], layers=(4, 12))
    with pytest.raises(SystemExit, match="2.npz holds other layers"):
        probe.load_states(tmp_path / "acts")
    with pytest.raises(SystemExit, match="holds no shard of hidden states"):
        probe.load_states(tmp_path / "none")


def prompt_row(key, name, condition, **more):
    row = {key: name, "prompt": f"prompt of {name}", "condition": condition, "options": OPTIONS}
    return row | {"letter_unknown": "C", "letter_correct": "C" if condition == "ambiguous" else "A"} | more


def test_the_evaluation_sets_are_read_with_the_stored_answers_of_both_models(tmp_path):
    rows = [
        prompt_row("id", "one.amb", "ambiguous", base_item="one", language="en"),
        prompt_row("id", "one.dis", "disambiguated", base_item="one", language="en"),
        prompt_row("id", "one.amb.told", "ambiguous", base_item="one", language="en", arm_tag="instruction"),
        prompt_row("id", "one.amb.long", "ambiguous", base_item="one", language="en", version="p1"),
    ]
    write_jsonl(tmp_path / "base.jsonl", [row | {"response": "B"} for row in rows])
    write_jsonl(tmp_path / "tuned.jsonl", [row | {"response": "C"} for row in rows])
    asked = [
        prompt_row("prompt_id", f"x.tr.{at}", ("ambiguous", "disambiguated")[at % 2], base_item="x.0") for at in (0, 1)
    ]
    write_jsonl(tmp_path / "prompts" / "tr.jsonl", [row | {"language": "tr"} for row in asked])
    for name, said in (("gpu_base.jsonl", "A"), ("gpu_tuned.jsonl", "C.")):
        answers = [{"prompt_id": row["prompt_id"], "response": said, "prompt": "as rendered"} for row in asked]
        write_jsonl(tmp_path / name, answers[::-1])
    bare = [prompt_row("id", "lone", "ambiguous")]
    write_jsonl(tmp_path / "lone.jsonl", [row | {"response": "A"} for row in bare])
    files = {"base": "base.jsonl", "fine_tuned": "tuned.jsonl"}
    spec = {
        "sets": [
            {"name": "plain", **files, "keep": [{"arm_tag": None, "version": None}]},
            {"name": "long", **files, "keep": [{"version": "p1"}, {"condition": "disambiguated"}]},
            {
                "name": "official_tr",
                "prompts": "prompts/tr.jsonl",
                "base": "gpu_base.jsonl",
                "fine_tuned": "gpu_tuned.jsonl",
            },
            {"name": "lone", "base": "lone.jsonl", "fine_tuned": "lone.jsonl"},
            {"name": "both", "of": ["official_tr", "plain"]},
        ]
    }
    (tmp_path / "sets.json").write_text(json.dumps(spec), encoding="utf-8")

    sets = probe.read_sets(tmp_path / "sets.json")

    assert list(sets) == ["plain", "long", "official_tr", "lone", "both"]
    assert [row["id"] for row in sets["plain"]] == ["one.amb", "one.dis"]
    assert [row["id"] for row in sets["long"]] == ["one.dis", "one.amb.long"]
    first = sets["plain"][0]
    assert first == {
        "id": "one.amb",
        "prompt": "prompt of one.amb",
        "condition": "ambiguous",
        "texts": {letter: f"option text {letter}" for letter in "ABC"},
        "letter_unknown": "C",
        "letter_correct": "C",
        "pair": "en/one",
        "cluster": "one",
        "responses": {"base": "B", "fine_tuned": "C"},
    }
    # prompt rows are joined with the answers of the responses dataset; a row without a base item is its own
    official = sets["official_tr"]
    assert [(row["id"], row["pair"], row["cluster"]) for row in official] == [
        ("x.tr.0", "tr/x.0", "x.0"),
        ("x.tr.1", "tr/x.0", "x.0"),
    ]
    assert official[1]["responses"] == {"base": "A", "fine_tuned": "C."}
    assert (sets["lone"][0]["pair"], sets["lone"][0]["cluster"]) == ("/lone", "lone")
    assert sets["both"] == [*official, *sets["plain"]]
    write_jsonl(tmp_path / "gpu_tuned.jsonl", [{"prompt_id": "x.tr.0", "response": "C"}])
    with pytest.raises(SystemExit, match="official_tr: x.tr.1 has no stored answer of fine_tuned"):
        probe.read_sets(tmp_path / "sets.json")


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Files in the shape of BBQ with their sha256 recorded, and the training set and test file built from them."""
    monkeypatch.setattr(bbq, "FILES", fake_bbq.write(tmp_path / "bbq"))
    benign = tmp_path / "benign.jsonl"
    benign.write_text("", encoding="utf-8")
    made = tmp_path / "data"
    assert build.main(["--bbq", str(tmp_path / "bbq"), "--benign", str(benign), "--out", str(made)], no_download) == 0
    return tmp_path


def test_the_probe_is_fitted_on_the_trained_items_and_chosen_on_whole_templates_of_them(data, capsys):
    notes = read_jsonl(data / "data" / FOLDER / "en.meta.jsonl")
    tested = read_jsonl(data / "data" / "test_en.jsonl")
    trained = {note["base_item"] for note in notes if note.get("kind") == "abstention"}

    rows = probe_rows.training(data / "bbq", data / "data")

    assert len(rows) == 2 * len(trained) and {row["base_item"] for row in rows} == trained
    assert Counter(row["label"] for row in rows) == {1: len(trained), 0: len(trained)}
    for open_, settled in zip(rows[::2], rows[1::2], strict=True):
        assert (open_["id"], settled["id"]) == (f"{open_['base_item']}.amb", f"{open_['base_item']}.dis")
        assert (open_["condition"], open_["label"], settled["label"]) == ("ambiguous", 1, 0)
        assert (open_["split"], open_["set"]) == (settled["split"], "train")
        # the two rows of an item share one option order, and the open row's answer is the unknown option
        assert [option["text"] for option in open_["options"]] == [option["text"] for option in settled["options"]]
        assert open_["letter_correct"] == open_["letter_unknown"] != settled["letter_correct"]
        assert open_["prompt"].endswith("\n" + build.LETTER_INSTRUCTION) and open_["prompt"] != settled["prompt"]
    fit, dev = ([row for row in rows if row["split"] == split] for split in ("fit", "dev"))
    # dev is whole templates, of every category, about a fifth of the items
    assert not {row["template"] for row in dev} & {row["template"] for row in fit}
    assert {row["category"] for row in dev} == {row["category"] for row in rows} == set(build.TRAINED_CATEGORIES)
    assert 0.1 < len(dev) / len(rows) < 0.3
    placed = Counter(row["letter_unknown"] for row in rows if row["label"])
    assert max(placed.values()) - min(placed.values()) <= 1
    assert not {row["base_item"] for row in rows} & {row["base_item"] for row in tested}
    assert not {row["template"] for row in rows} & {row["template"] for row in tested}
    assert probe_rows.training(data / "bbq", data / "data") == rows

    # the evaluation prompts: every distinct prompt once, under the first set that holds it
    plain = [row for row in tested if not row.get("arm_tag") and row["format"] == "letter" and row["order"] == 0][:6]
    write_jsonl(data / "base.jsonl", [row | {"id": row["prompt_id"], "response": "A"} for row in plain])
    sets = {"sets": [{"name": "test", "base": "base.jsonl", "fine_tuned": "base.jsonl"}]}
    sets["sets"] += [
        {"name": "again", "base": "base.jsonl", "fine_tuned": "base.jsonl"},
        {"name": "all", "of": ["test"]},
    ]
    (data / "sets.json").write_text(json.dumps(sets), encoding="utf-8")
    asked = ["--bbq", str(data / "bbq"), "--data", str(data / "data"), "--sets", str(data / "sets.json")]

    assert probe_rows.main([*asked, "--out", str(data / "out")], no_download) == 0

    assert read_jsonl(data / "out" / "probe_train_in.jsonl") == rows
    held = read_jsonl(data / "out" / "probe_eval_in.jsonl")
    assert held == [
        {"id": f"test|{row['prompt_id']}", "prompt": row["prompt"], "set": "test", "source_id": row["prompt_id"]}
        for row in plain
    ]
    summary = json.loads((data / "out" / "probe_build.json").read_text(encoding="utf-8"))
    assert (summary["train_rows"], summary["fit_base_items"] + summary["dev_base_items"]) == (len(rows), len(trained))
    assert summary["evaluation_rows"] == {"test": 6} and set(summary["overlap"].values()) == {0}
    assert summary["dev_templates"] + summary["fit_templates"] == len({row["template"] for row in rows})
    for name, digest in summary["files_sha256"].items():
        assert hashlib.sha256((data / "out" / name).read_bytes()).hexdigest() == digest
    assert '"train_rows"' in capsys.readouterr().out
    # a training prompt among the evaluation prompts stops the step before anything is written
    write_jsonl(data / "base.jsonl", [rows[0] | {"response": "A"}])
    with pytest.raises(SystemExit, match="not apart from what they are tested on"):
        probe_rows.main([*asked, "--out", str(data / "leaky")], no_download)
    assert not (data / "leaky").exists()


def test_the_choice_takes_the_best_dev_auc_then_the_lower_loss_the_smaller_c_and_the_lower_layer():
    def tried(layer, c, auc, loss):
        return {"layer": layer, "C": c, "dev_auc": auc, "dev_log_loss": loss, "fit_auc": 1.0}

    assert probe_fit.choose([tried(4, 1.0, 0.9, 0.1), tried(8, 0.1, 0.95, 0.5)])["layer"] == 8
    assert probe_fit.choose([tried(4, 1.0, 0.95, 0.3), tried(8, 0.1, 0.95, 0.2)])["layer"] == 8
    assert probe_fit.choose([tried(8, 1.0, 0.95, 0.2), tried(12, 0.1, 0.95, 0.2)])["layer"] == 12
    assert probe_fit.choose([tried(12, 0.1, 0.95, 0.2), tried(8, 0.1, 0.95, 0.2)])["layer"] == 8

    def at(threshold, hit, false):
        return {"threshold": threshold, "true_positive_rate": hit, "false_positive_rate": false}

    # the widest gap between the two rates; then the threshold nearest one half; then the lower one
    assert probe_fit.threshold_of([at(0.1, 1.0, 0.2), at(0.3, 0.95, 0.05), at(0.5, 0.8, 0.0)])["threshold"] == 0.3
    assert probe_fit.threshold_of([at(0.1, 1.0, 0.0), at(0.4, 1.0, 0.0), at(0.9, 1.0, 0.0)])["threshold"] == 0.4
    assert probe_fit.threshold_of([at(0.6, 1.0, 0.0), at(0.4, 1.0, 0.0)])["threshold"] == 0.4
    labels, scores = np.array([1, 1, 0, 0]), np.array([0.9, 0.3, 0.3, 0.02])
    swept = probe_fit.sweep(labels, scores)
    assert [row["threshold"] for row in swept] == list(probe.THRESHOLDS) and len(swept) == 19
    assert swept[0] == at(0.05, 1.0, 0.5) and swept[5] == at(0.3, 1.0, 0.5) and swept[6] == at(0.35, 0.5, 0.0)


def training_rows(count):
    """Rows of ``count`` prompts, half of them open; every fourth pair is dev."""
    return [
        {"id": f"row-{at}", "label": at % 2, "split": "dev" if (at // 2) % 4 == 0 else "fit"} for at in range(count)
    ]


def test_the_fit_reads_the_fit_rows_picks_on_dev_and_freezes_arrays_that_read_dev_as_the_pipeline_did(tmp_path, capsys):
    rows = training_rows(240)
    told, labels = separable(240)
    assert [row["label"] for row in rows] == labels.tolist()
    noise = np.random.default_rng(1).normal(size=told.shape).astype(np.float32)
    # the first layer holds nothing of the label, the second holds it
    states = np.stack([noise, told], axis=1)
    half = len(rows) // 2
    shard(tmp_path / "acts", 0, [row["id"] for row in rows[:half]], states[:half])
    shard(tmp_path / "acts", 1, [row["id"] for row in rows[half:]], states[half:])
    write_jsonl(tmp_path / "rows.jsonl", rows)
    asked = ["--rows", str(tmp_path / "rows.jsonl"), "--states", str(tmp_path / "acts")]

    assert probe_fit.main([*asked, "--out", str(tmp_path / "out")]) == 0

    record = json.loads((tmp_path / "out" / "probe_fit.json").read_text(encoding="utf-8"))
    dev = [at for at, row in enumerate(rows) if row["split"] == "dev"]
    assert (record["fit_rows"], record["dev_rows"]) == (len(rows) - len(dev), len(dev)) and len(dev) == 60
    assert [(row["layer"], row["C"]) for row in record["grid"]] == [
        (layer, c) for layer in (4, 8) for c in probe.C_GRID
    ]
    assert record["selected"]["layer"] == 8 and record["selected"]["dev_auc"] == 1.0
    assert record["selected"] == probe_fit.choose(record["grid"])
    assert max(row["dev_auc"] for row in record["grid"] if row["layer"] == 4) < 0.8
    assert record["threshold"] == probe_fit.threshold_of(record["threshold_sweep"])
    assert (record["dev_ambiguous_rows"], record["dev_disambiguated_rows"]) == (30, 30)
    assert (record["dev_fired_on_ambiguous"], record["dev_fired_on_disambiguated"]) == (30, 0)
    frozen = probe.load(tmp_path / "out")
    assert (frozen.layer, frozen.c, frozen.threshold) == (8, record["selected"]["C"], record["threshold"]["threshold"])
    # the frozen arrays are those of a fit on the fit rows alone
    fit = [at for at, row in enumerate(rows) if row["split"] == "fit"]
    again = make_pipeline(StandardScaler(), LogisticRegression(C=frozen.c, max_iter=5000)).fit(told[fit], labels[fit])
    assert np.allclose(frozen.probability(told[dev]), again.predict_proba(told[dev])[:, 1], atol=1e-6)
    assert np.array_equal(frozen.fires(told[dev]), labels[dev] == 1)
    settings = json.loads((tmp_path / "out" / "probe.json").read_text(encoding="utf-8"))
    assert (settings["layers_extracted"], settings["fitted_on"]) == ([4, 8], "rows.jsonl")
    assert settings["chosen_from"]["C"] == list(probe.C_GRID)
    assert '"selected"' in capsys.readouterr().out and "grid" not in capsys.readouterr().out
    write_jsonl(tmp_path / "rows.jsonl", rows[:-1])
    with pytest.raises(SystemExit, match="240 hidden states for 239 training rows"):
        probe_fit.main([*asked, "--out", str(tmp_path / "short")])


class Tokenizer:
    """A token per character after one BOS; a batch is padded on the left with 0."""

    bos_token_id = 1

    def __init__(self, bos=1, left=True):
        self.bos, self.left = bos, left

    def _ids(self, text):
        return [1] * self.bos + [2 + ord(char) % 50 for char in text]

    def __call__(self, text, return_tensors=None, padding=False):
        if isinstance(text, str):
            return {"input_ids": self._ids(text)}
        rows = [self._ids(one) for one in text]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row if self.left else row + [0] * (width - len(row)) for row in rows]
        held = torch.tensor(padded)
        return {"input_ids": held, "attention_mask": (held != 0).long()}


class Model:
    """Hidden state per layer and position: the layer, the token there, and how many tokens came before it."""

    def __init__(self, layers=4):
        self.config = SimpleNamespace(hidden_size=3, num_hidden_layers=layers)
        self.returned, self.batches = layers + 1, []

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        self.batches.append(tuple(input_ids.shape))
        seen = attention_mask.cumsum(dim=1)
        hidden = [
            torch.stack([torch.full_like(input_ids, layer), input_ids, seen], dim=-1).float() for layer in range(5)
        ]
        return SimpleNamespace(hidden_states=tuple(hidden[: self.returned]))


def test_the_state_at_the_last_token_is_taken_for_each_prompt_whatever_its_batch():
    rendered = ["a longer prompt", "tiny", "middle one", "x", "the longest prompt of all"]
    model, said = Model(), []

    states, lengths = probe_extract.last_token_states(
        model, Tokenizer(), rendered, [2, 4], batch_size=2, said=said.append
    )

    assert states.shape == (5, 2, 3) and states.dtype == np.float32
    assert lengths == [1 + len(text) for text in rendered]
    for at, text in enumerate(rendered):
        last = 2 + ord(text[-1]) % 50
        assert states[at].tolist() == [[2, last, lengths[at]], [4, last, lengths[at]]]
    # batches go by length, the shortest prompts first
    assert model.batches == [(2, 5), (2, 16), (1, 26)] and said == [2, 4, 5]
    with pytest.raises(SystemExit, match="does not start with exactly one BOS"):
        probe_extract.last_token_states(Model(), Tokenizer(bos=2), rendered, [2])
    with pytest.raises(SystemExit, match="padding is not on the left"):
        probe_extract.last_token_states(Model(), Tokenizer(left=False), rendered, [2])
    short = Model()
    short.returned = 4
    with pytest.raises(SystemExit, match="the model returned 4 hidden states"):
        probe_extract.last_token_states(short, Tokenizer(), rendered, [2])


def test_more_processes_than_the_thread_cap_allows_are_refused(tmp_path, capsys):
    asked = ["--model", "m", "--manifest", "x", "--input", "in", "--output", str(tmp_path / "acts")]
    with pytest.raises(SystemExit):
        probe_extract.main([*asked, "--workers", "8", "--threads", "10"])
    assert "over the thread cap" in capsys.readouterr().err and not (tmp_path / "acts").exists()
    # a shard that is already written is not computed again, and no model is loaded for it
    shard(tmp_path / "acts", 0, ["a"], np.zeros((1, 2, 3), dtype=np.float32))
    assert probe_extract.main(asked) == 0 and "already written" in capsys.readouterr().out


def test_the_scores_put_the_gate_on_stored_answers_and_write_no_text(tmp_path, capsys):
    told, labels = separable(40, seed=9)
    # a probe that reads the third unit: it fires on the open rows
    weights = np.zeros(6, dtype=np.float32)
    weights[2] = 3.0
    frozen = probe.Probe(8, 0.1, 0.5, np.zeros(6, np.float32), np.ones(6, np.float32), weights, np.zeros(1, np.float32))
    probe.save(frozen, tmp_path / "frozen")
    asked, answers = [], {"base": [], "fine_tuned": []}
    for at, label in enumerate(labels.tolist()):
        condition = "ambiguous" if label else "disambiguated"
        row = prompt_row("id", f"item-{at // 2}.{condition}", condition, base_item=f"item-{at // 2}", language="en")
        asked.append({"id": f"first|{row['id']}", "prompt": row["prompt"]})
        answers["base"].append(row | {"response": "A. option text A" if at % 4 < 2 else "the first one"})
        answers["fine_tuned"].append(row | {"response": row["letter_correct"]})
    for name, rows in answers.items():
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    sets = {"sets": [{"name": "first", "base": "base.jsonl", "fine_tuned": "fine_tuned.jsonl"}]}
    sets["sets"].append({"name": "whole", "of": ["first"]})
    (tmp_path / "sets.json").write_text(json.dumps(sets), encoding="utf-8")
    write_jsonl(tmp_path / "eval_in.jsonl", asked)
    states = np.stack([np.zeros_like(told), told], axis=1)
    shard(tmp_path / "acts", 0, [row["id"] for row in asked], states)
    paths = {"--probe": "frozen", "--sets": "sets.json", "--prompts": "eval_in.jsonl", "--states": "acts"}
    command = [part for flag, name in paths.items() for part in (flag, str(tmp_path / name))]

    assert probe_scores.main([*command, "--out", str(tmp_path / "results"), "--fine-tune", "defense"]) == 0

    names = ["probe-defense-answers.jsonl", "probe-defense.json", "probe-defense.md"]
    assert sorted(path.name for path in (tmp_path / "results").iterdir()) == names
    tables = ProbeTables.model_validate_json((tmp_path / "results" / "probe-defense.json").read_bytes())
    assert tables.sets == ["first", "whole"] and tables.probe.layer == 8 and tables.seeds.seed == probe.SEED
    rates = {(row.set, row.system, row.metric): (row.k, row.n) for row in tables.rates}
    # the base model names a person on half of the open rows and is unread on the other half; the gate fires on all
    assert rates["first", "base", "names_a_person"] == (10, 20) and rates["first", "base", "unparseable"] == (20, 40)
    assert rates["first", "base_plus_probe", "names_a_person"] == (0, 20)
    assert rates["first", "base_plus_probe", "twin_both_right"] == (10, 20)
    assert rates["first", "fine_tuned", "wrong_answers"] == (0, 40) and rates["whole", "base", "wrong_answers"] == (
        30,
        40,
    )
    reading = tables.readings[0]
    assert (reading.auc, reading.fired_on_ambiguous, reading.fired_on_disambiguated) == (1.0, 20, 0)
    written = read_jsonl(tmp_path / "results" / "probe-defense-answers.jsonl")
    # one row per prompt of the set read from files; the set made of it adds none
    assert len(written) == 40 and {row["set"] for row in written} == {"first"}
    assert written[1] == {
        "set": "first",
        "id": "item-0.ambiguous",
        "condition": "ambiguous",
        "probe_score": round(float(frozen.probability(told[1:2])[0]), 6),
        "fired": True,
        "letter_base": "A",
        "letter_base_plus_probe": "C",
        "letter_fine_tuned": "C",
        "letter_unknown": "C",
        "letter_correct": "C",
    }
    shown = (tmp_path / "results" / "probe-defense.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == shown + "\n"
    assert "| first | 50.0% (10/20) | 0.0% (0/20) | 0.0% (0/20) |" in shown and "attribution fine-tune" in shown
    # no prompt, option or answer text reaches a file
    public = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "results").iterdir())
    assert "prompt of" not in public and "option text" not in public and "the first one" not in public

    # without --fine-tune the files carry no fine-tune's name
    assert probe_scores.main([*command, "--out", str(tmp_path / "plain")]) == 0
    assert sorted(path.name for path in (tmp_path / "plain").iterdir()) == [
        "probe-answers.jsonl",
        "probe.json",
        "probe.md",
    ]
    # a prompt without a hidden state stops the step
    write_jsonl(tmp_path / "eval_in.jsonl", asked[:-1])
    shard(tmp_path / "fewer", 0, [row["id"] for row in asked[:-1]], states[:-1])
    fewer = [str(tmp_path / "fewer") if part == str(tmp_path / "acts") else part for part in command]
    with pytest.raises(SystemExit, match="1 prompts of the evaluation sets have no hidden state"):
        probe_scores.main([*fewer, "--out", str(tmp_path / "none")])
    with pytest.raises(SystemExit, match="40 hidden states of layers .* for 39 prompts and layer 8"):
        probe_scores.main([*command, "--out", str(tmp_path / "none")])
    assert not (tmp_path / "none").exists()
