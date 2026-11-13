from enum import IntEnum

import pytest

from driftmap.core import schemas
from driftmap.core.schemas import DriftmapError, ExitCode

EXIT_CODES = {
    "SUCCESS": 0,
    "UNEXPECTED": 1,
    "USAGE": 2,
    "READINESS": 3,
    "GUARD_REFUSAL": 4,
    "LOST_RUN": 5,
    "REDACTION": 6,
    "PUSH_FAILED": 7,
    "DRIFT_BELOW_THRESHOLD": 8,
    "PARTIAL": 10,
    "OOM": 11,
    "INTEGRITY": 12,
    "COLUMN_FAILED": 13,
    "JUDGE_INSPECT_STOP": 14,
}

SCHEMA_IDS = {
    "run": "driftmap.run/1",
    "decision": "driftmap.decision/1",
    "ledger": "driftmap.ledger/1",
    "plan": "driftmap.plan/1",
    "generation": "driftmap.generation/1",
    "label": "driftmap.label/1",
    "alpha_screen": "driftmap.alpha_screen/1",
    "gguf": "driftmap.gguf/1",
    "manifest": "driftmap.manifest/1",
    "tables": "driftmap.tables/1",
}


def test_exit_code_members():
    assert issubclass(ExitCode, IntEnum)
    assert {m.name: int(m) for m in ExitCode} == EXIT_CODES
    assert len(ExitCode) == len(EXIT_CODES) == 14
    assert 9 not in {int(m) for m in ExitCode}


def test_exit_code_is_an_int():
    assert ExitCode.INTEGRITY == 12
    assert isinstance(ExitCode(12), int)
    with pytest.raises(ValueError):
        ExitCode(9)


def test_driftmap_error_carries_exit_code_and_message():
    err = DriftmapError(12, "merged hash mismatch")
    assert err.exit_code == 12
    assert isinstance(err.exit_code, ExitCode)
    assert err.message == "merged hash mismatch"
    assert str(err) == "merged hash mismatch"
    assert isinstance(err, Exception)


def test_driftmap_error_accepts_enum_and_defaults_message():
    err = DriftmapError(ExitCode.GUARD_REFUSAL)
    assert err.exit_code is ExitCode.GUARD_REFUSAL
    assert err.message == ""
    assert err.args == (ExitCode.GUARD_REFUSAL, "")


def test_driftmap_error_rejects_unknown_code():
    with pytest.raises(ValueError):
        DriftmapError(9, "no such code")


def test_driftmap_error_subclass_pins_its_code():
    class TokenizerMismatch(DriftmapError):
        exit_code_default = ExitCode.INTEGRITY

    err = TokenizerMismatch("canary row differs")
    assert err.exit_code == 12
    assert str(err) == "canary row differs"
    assert isinstance(err, DriftmapError)
    assert TokenizerMismatch(ExitCode.INTEGRITY, "agreeing code").exit_code == 12
    assert TokenizerMismatch().message == ""


def test_pinned_subclass_rejects_a_disagreeing_code_or_a_second_positional():
    class Partial(DriftmapError):
        exit_code_default = ExitCode.PARTIAL

    with pytest.raises(TypeError):
        Partial(ExitCode.OOM, "wrong code")
    with pytest.raises(TypeError):
        Partial(11, "wrong code")
    with pytest.raises(TypeError):
        Partial("message", "second positional")


def test_base_error_rejects_a_message_in_the_code_slot():
    with pytest.raises(TypeError):
        DriftmapError("merged hash mismatch")
    with pytest.raises(TypeError):
        DriftmapError()


def test_message_is_always_a_string():
    cause = ValueError("shard 3 missing")
    err = DriftmapError(ExitCode.INTEGRITY, cause)
    assert err.message == "shard 3 missing"
    assert isinstance(err.message, str)
    assert str(err) == "shard 3 missing"
    assert DriftmapError(1, None).message == "None"
    assert DriftmapError(1, 404).message == "404"


def test_schema_id_constants():
    for short, schema_id in SCHEMA_IDS.items():
        assert getattr(schemas, f"SCHEMA_{short.upper()}") == schema_id


def test_schema_versions_lists_every_schema_once():
    versions = schemas.schema_versions()
    assert versions == SCHEMA_IDS
    assert len(set(versions.values())) == len(versions)
    assert all(v.startswith("driftmap.") and v.endswith("/1") for v in versions.values())


def test_schema_versions_returns_a_fresh_copy():
    first = schemas.schema_versions()
    first["run"] = "tampered"
    assert schemas.schema_versions()["run"] == SCHEMA_IDS["run"]
