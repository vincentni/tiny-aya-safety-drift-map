import random

import pytest

from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.judge.incremental import (
    RawRow,
    counts,
    dump_rows,
    judged_rows,
    pending,
    read_rows,
    rebuilt,
    union,
)
from driftmap.judge.rubric import ParsedLabels, judge_config, parse
from tests.fakes.judge_world import CANONICAL, SAFE, UNPARSEABLE, generation_rows
from tests.fakes.settings import settings_toml


@pytest.fixture
def config(tmp_path):
    return judge_config(load_settings(settings_toml(tmp_path)))


def judged(rows, config, raw=SAFE):
    return judged_rows(rows, [(raw, parse(raw))] * len(rows), config)


def test_a_label_file_grows_by_union_and_a_second_pass_changes_no_byte(config):
    """40 stored rows and 60 new ones make 100 keyed rows; writing them again gives the same bytes (TS-JDG-5)."""
    rows = generation_rows("harmful", "en", 100)
    existing, raws = judged(rows[:40], config)
    new, new_raws = judged(rows[40:], config, CANONICAL)
    raws += new_raws

    merged = union(existing, new)

    assert len(merged) == 100 and len({(row.checkpoint, row.language, row.prompt_id) for row in merged}) == 100
    assert dump_rows(union(merged, new)) == dump_rows(merged) == dump_rows(union(merged, []))
    shuffled = random.Random(7).sample(new, len(new))
    assert dump_rows(union(existing, shuffled)) == dump_rows(merged)
    assert dump_rows(merged).count(b"\n") == 100 and dump_rows(merged).endswith(b"}\n")
    again, _ = judged(rows[:1], config, CANONICAL)
    assert union(merged, again)[0].response_harm == "yes" != merged[0].response_harm
    assert pending(rows, merged, config.config_hash, raws=raws) == []
    assert pending(rows, existing, config.config_hash, raws=raws) == rows[40:]
    assert pending(rows, merged, "another config", raws=raws) == rows


def test_rows_without_a_response_are_labelled_unscored_and_never_rejudged(config):
    rows = generation_rows("harmful", "fr", 3)
    rows[1] = rows[1].model_copy(update={"response": None, "status": "unscored", "error": "empty"})
    long_raw = UNPARSEABLE[2] + "x" * 5000

    labels, raws = judged_rows(rows, [(SAFE, parse(SAFE)), (long_raw, None)], config)

    assert [(row.response_harm, row.raw_output_present) for row in labels] == [
        ("no", True),
        ("unscored", False),
        ("unscored", True),
    ]
    assert labels[2].request_harm == labels[2].refusal == "unscored" and labels[0].judge == config
    assert labels[0].prompt == rows[0].prompt and labels[0].chrf is None and labels[0].set == "harmful"
    assert [raw.prompt_id for raw in raws] == [rows[0].prompt_id, rows[2].prompt_id]
    assert len(raws[1].raw) == 2000 and raws[1].raw.startswith(UNPARSEABLE[2])
    assert (raws[0].repo, raws[0].revision) == (config.repo, config.revision)
    assert pending(rows, labels, config.config_hash, rejudge_unscored=True, raws=raws) == [rows[2]]
    assert pending(rows, labels, config.config_hash, raws=raws) == []
    assert counts(labels)["refusal"] == {"yes": 0, "no": 1, "unscored": 2}
    with pytest.raises(DriftmapError, match="1 answers for 2 responses"):
        judged_rows(rows, [(SAFE, ParsedLabels("no", "no", "no"))], config)


def test_a_label_counts_only_while_its_stored_output_says_the_same(config):
    rows = generation_rows("harmful", "es", 6)
    labels, raws = judged(rows, config)
    flipped, flipped_raws = judged(rows[:1], config, CANONICAL)
    other_judge = raws[1].model_copy(update={"revision": "9" * 40})
    other_response = raws[2].model_copy(update={"response_sha256": "0" * 64})
    unparsed = raws[3].model_copy(update={"raw": UNPARSEABLE[2]})
    refused = raws[5].model_copy(update={"raw": SAFE.replace('"refusal": "no"', '"refusal": "yes"')})
    asked = raws[5].model_copy(update={"raw": SAFE.replace('"request_harm": "no"', '"request_harm": "yes"')})
    stored = [flipped_raws[0], other_judge, other_response, unparsed, refused]

    # a newer output with another verdict, another judge, another response, a lost parse, no output at all
    assert pending(rows, labels, config.config_hash, raws=stored) == rows
    assert pending(rows, labels, config.config_hash, raws=[*raws[:5], asked]) == rows[5:]
    assert pending(rows, labels, config.config_hash, raws=raws[:4]) == rows[4:]
    assert pending(rows, union(labels, flipped), config.config_hash, raws=[*flipped_raws, *raws[1:]]) == []
    lost = rows[5].model_copy(update={"response": None})
    assert pending([lost], labels, config.config_hash, raws=raws) == [lost]
    unscored, unscored_raws = judged(rows[:1], config, UNPARSEABLE[2])
    assert pending(rows[:1], unscored, config.config_hash, raws=unscored_raws) == []
    # the same response under another rendered prefix: the label would carry the earlier prefix hash
    rendered = [rows[0], rows[1].model_copy(update={"rendered_prefix_sha256": "e" * 64})]
    assert pending(rendered, labels, config.config_hash, raws=raws) == rendered[1:]
    silent = lost.model_copy(update={"rendered_prefix_sha256": "e" * 64})
    earlier, _ = judged_rows([lost], [], config)
    assert pending([lost], earlier, config.config_hash, raws=[]) == []
    assert pending([silent], earlier, config.config_hash, raws=[]) == [silent]


def test_a_missing_label_is_rebuilt_from_a_stored_answer_of_the_same_judge_and_response(config):
    rows = generation_rows("harmful", "en", 6)
    labels, raws = judged(rows, config, CANONICAL)
    other_judge = raws[1].model_copy(update={"revision": "9" * 64})
    other_response = raws[2].model_copy(update={"response_sha256": "0" * 64})
    unparsed = raws[3].model_copy(update={"raw": UNPARSEABLE[0]})

    # no answer is stored for the fifth row, and the sixth still has its label
    found = rebuilt(rows, labels[5:], [raws[0], other_judge, other_response, unparsed, raws[5]], config)

    assert found == [labels[0]]
    assert rebuilt([rows[0].model_copy(update={"response": None})], [], raws, config) == []
    # a label of another config is asked again, never rebuilt
    earlier = labels[0].model_copy(update={"judge": config.model_copy(update={"config_hash": "earlier"})})
    assert rebuilt(rows[:1], [earlier], raws, config) == []
    renamed = config.model_copy(update={"repo": "another-model"})
    assert rebuilt(rows[:1], [], raws, renamed) == []


def test_rows_are_read_at_newlines_only(tmp_path):
    text = f"a{chr(0x2028)}b{chr(0x85)}c"
    row = RawRow(checkpoint="c", language="en", prompt_id="p", repo="r", revision="v", response_sha256="0", raw=text)
    path = tmp_path / "raw.jsonl"
    path.write_bytes(b"\n" + dump_rows([row, row]) + b"\n")

    assert read_rows(path, RawRow) == [row, row] and read_rows(tmp_path / "absent.jsonl", RawRow) == []
    path.write_bytes(dump_rows([row]) + b'{"checkpoint": "c"}\n')
    with pytest.raises(DriftmapError, match=r"raw\.jsonl:2: not a RawRow \(language: Field required\)") as caught:
        read_rows(path, RawRow)
    assert caught.value.exit_code == 12
