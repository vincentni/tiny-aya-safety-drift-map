from dataclasses import replace
from typing import get_args

import pytest

from driftmap.core.records import JobKind
from driftmap.jobs.kinds import KINDS, State, prerequisites

LANGUAGES = ("en", "fr", "es", "ru", "hi", "ar")
READY = State(
    languages=LANGUAGES,
    checkpoint="drifted",
    merged_sha256="ab" * 32,
    responses_complete=True,
    baseline_labels=frozenset(LANGUAGES),
    screen_responses=True,
    repaired_revision="0123456789abcdef0123456789abcdef01234567",
    gguf_record=True,
    recover_status="failed",
)


def test_table_covers_the_seven_kinds():
    assert set(KINDS) == set(get_args(JobKind))
    assert {kind for kind, row in KINDS.items() if not row.gpu} == {"quant"}
    assert all(row.module == f"driftmap.scripts.{kind}" and row.image == "pytorch" for kind, row in KINDS.items())
    assert {kind for kind, row in KINDS.items() if "model" in row.reads} == {"drift", "generate", "repair"}
    assert not any("base" in row.reads for row in KINDS.values())
    assert KINDS["quant"].resume_key is None
    assert KINDS["generate"].resume_key == "(set, language)"


@pytest.mark.parametrize(
    ("kind", "missing", "named"),
    [
        ("generate", {"merged_sha256": None}, "merged_sha256"),
        ("judge", {"responses_complete": False}, "responses"),
        ("repair", {"baseline_labels": frozenset(LANGUAGES[:4])}, "baseline labels for hi, ar"),
        ("repair", {"languages": ()}, "baseline labels"),
        ("repair", {"screen_responses": False}, "screen"),
        ("quant", {"repaired_revision": None}, "repaired"),
        ("quant_eval", {"gguf_record": False}, "gguf.json"),
    ],
)
def test_missing_input_is_named(kind, missing, named):
    assert prerequisites(kind, READY) == []
    found = prerequisites(kind, replace(READY, **missing))
    assert len(found) == 1
    assert named in found[0]


def test_nothing_on_the_hub_blocks_all_but_drift_and_baseline():
    empty = State(languages=LANGUAGES)
    assert prerequisites("drift", empty) == []
    assert prerequisites("generate", empty) == []
    assert len(prerequisites("repair", empty)) == 2
    assert all(prerequisites(kind, empty) for kind in ("judge", "repair", "quant", "quant_eval", "steer"))
    with pytest.raises(KeyError):
        prerequisites("train", empty)


def test_steer_needs_a_failed_recovery_or_the_comparison_flag():
    """Steer runs after a failed recovery or with the comparison flag, never otherwise (TS-LCH-12)."""
    assert prerequisites("steer", replace(READY, recover_status="failed")) == []
    assert prerequisites("steer", replace(READY, recover_status="chosen", steer_comparison=True)) == []
    for state in (replace(READY, recover_status="chosen"), replace(READY, recover_status=None)):
        (missing,) = prerequisites("steer", state)
        assert "recover_status" in missing
        assert "comparison" in missing
