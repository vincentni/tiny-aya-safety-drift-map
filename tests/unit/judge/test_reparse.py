import subprocess
import sys

import pytest

from driftmap.core.hub import Hub
from driftmap.core.ids import labels_path
from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.judge import rubric
from driftmap.judge.incremental import dump_rows, pending
from driftmap.judge.reparse import reparse, reparse_checkpoint, reparse_rows
from tests.fakes.judge_world import CANONICAL, CHECKPOINT, EVAL, RESPONSES, SLUG, UNPARSEABLE, JudgeWorld

N_A = UNPARSEABLE[2]


@pytest.fixture
def lenient(monkeypatch):
    """A later parser that reads a refusal value of N/A as no."""
    strict = rubric.parse
    monkeypatch.setattr(rubric, "PARSER_VERSION", "2")
    monkeypatch.setattr(rubric, "parse", lambda raw: strict(raw.replace('"refusal": "N/A"', '"refusal": "no"')))


def seeded(tmp_path, monkeypatch, special=None):
    world = JudgeWorld(tmp_path, monkeypatch, languages=("en",))
    world.seed(rubric.judge_config(load_settings(world.config)), special=special)
    return world


def test_a_parser_change_is_applied_to_the_stored_answers_without_a_model(tmp_path, monkeypatch, request):
    """A new parser version rewrites the label file under its own config hash and keeps the earlier one (TS-JDG-8)."""
    world = seeded(tmp_path, monkeypatch, {"harmful-en-000": N_A, "harmful-en-001": N_A})
    before, old_file = world.labels(), world.stored(labels_path(SLUG, "en", "full"), EVAL)
    raw_head = world.fake.heads[RESPONSES]
    assert [row.response_harm for row in before[:3]] == ["unscored", "unscored", "no"]
    hub = Hub(world.fake)
    assert reparse_checkpoint(hub, EVAL, RESPONSES, CHECKPOINT) == []
    request.getfixturevalue("lenient")

    old_hash = before[0].judge.config_hash
    kept = f"labels/{SLUG}/superseded/{old_hash}/en.jsonl"
    assert len(old_hash) == 64
    assert reparse_checkpoint(hub, EVAL, RESPONSES, CHECKPOINT) == [kept, labels_path(SLUG, "en", "full")]

    after = world.labels()
    new_config = rubric.judge_config(load_settings(world.config))
    assert new_config.parser_version == "2" and new_config.config_hash != old_hash
    assert all(row.judge == new_config for row in after)
    assert [(row.request_harm, row.refusal, row.response_harm) for row in after[:3]] == [
        ("yes", "no", "yes"),
        ("yes", "no", "yes"),
        ("no", "no", "no"),
    ]
    same = [row.model_copy(update={"judge": new_config}) for row in before[2:]]
    assert after[2:] == same and len(after) == len(before) == 8
    assert world.stored(kept, EVAL) == old_file and world.fake.heads[RESPONSES] == raw_head
    assert reparse_checkpoint(hub, EVAL, RESPONSES, CHECKPOINT) == []
    assert reparse_checkpoint(hub, EVAL, RESPONSES, CHECKPOINT, mode="smoke") == []
    # a judge run under the new parser finds every row labelled
    rows = [row for unit in world.rows.values() for row in unit]
    assert pending(rows, after, new_config.config_hash, raws=world.raws()) == []


def test_a_judged_row_whose_answer_was_not_stored_cannot_be_reparsed(tmp_path, monkeypatch, lenient):
    monkeypatch.setattr(rubric, "PARSER_VERSION", "1")
    world = seeded(tmp_path, monkeypatch)
    monkeypatch.setattr(rubric, "PARSER_VERSION", "2")
    labels, raws = world.labels(), world.raws()

    with pytest.raises(DriftmapError, match="en/harmful-en-000: no stored judge output") as caught:
        reparse_rows(labels, raws[1:])
    assert caught.value.exit_code == 12
    other = raws[0].model_copy(update={"repo": "someone/other-judge"})
    with pytest.raises(DriftmapError, match="an output of someone/other-judge@"):
        reparse_rows(labels, [other, *raws[1:]])
    unanswered = labels[0].model_copy(update={"raw_output_present": False, "response_harm": "unscored"})
    fresh = reparse_rows([unanswered], [])
    assert (fresh[0].response_harm, fresh[0].judge.parser_version) == ("unscored", "2")


def test_rows_of_a_prompt_format_this_code_cannot_read_are_left_as_they_are(tmp_path, monkeypatch, lenient):
    monkeypatch.setattr(rubric, "PARSER_VERSION", "1")
    world = seeded(tmp_path, monkeypatch)
    monkeypatch.setattr(rubric, "PARSER_VERSION", "2")
    labels = world.labels()
    earlier = labels[0].judge.model_copy(update={"prompt_format": "polyguard-v1"})
    foreign = [row.model_copy(update={"judge": earlier}) for row in labels[:2]]
    unjudged = labels[2].model_copy(update={"judge": None})

    # no stored answer is needed for a row that is not read again
    assert reparse_rows([*foreign, unjudged], []) == [*foreign, unjudged]
    fresh = reparse_rows([*foreign, *labels[3:]], world.raws())
    assert fresh[:2] == foreign and {row.judge.parser_version for row in fresh[2:]} == {"2"}


@pytest.mark.parametrize("forged", ["../../../zzz", "a" * 63, "A" * 64, "a" * 64 + "/../zzz", ""])
def test_a_label_file_with_a_config_hash_that_is_no_hash_is_refused(tmp_path, monkeypatch, lenient, forged):
    monkeypatch.setattr(rubric, "PARSER_VERSION", "1")
    world = seeded(tmp_path, monkeypatch, {"harmful-en-000": CANONICAL})
    monkeypatch.setattr(rubric, "PARSER_VERSION", "2")
    labels = [
        row.model_copy(update={"judge": row.judge.model_copy(update={"config_hash": forged})}) for row in world.labels()
    ]
    target = tmp_path / "labels" / SLUG / "en.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(dump_rows(labels))
    (tmp_path / "raws" / SLUG).mkdir(parents=True)
    (tmp_path / "raws" / SLUG / "en.jsonl").write_bytes(dump_rows(world.raws()))

    with pytest.raises(DriftmapError, match="config hash") as caught:
        reparse(tmp_path / "labels", tmp_path / "raws")
    assert caught.value.exit_code == 12
    assert (
        sorted(path.name for path in tmp_path.rglob("*") if path.is_dir() and path.name in ("zzz", "superseded")) == []
    )
    assert target.read_bytes() == dump_rows(labels)


def test_reparse_imports_no_model_library(child_env):
    code = "import sys, driftmap.judge.reparse; sys.exit(bool({'torch', 'transformers', 'litellm'} & set(sys.modules)))"

    assert subprocess.run([sys.executable, "-c", code], check=False, env=child_env, timeout=100).returncode == 0
