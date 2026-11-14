import inspect
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from huggingface_hub import JobStage
from pydantic import BaseModel, TypeAdapter, ValidationError

from driftmap.core import records
from driftmap.core.records import (
    Ceilings,
    Decoding,
    DeltaRow,
    JudgeConfig,
    LedgerLine,
    RateRow,
    Ref,
    Refusal,
    RunRecord,
    Snapshot,
    Tables,
    dump,
    load,
)
from driftmap.core.schemas import schema_versions

SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "ab" * 32
IST = timezone(timedelta(hours=5, minutes=30))
LOCAL = datetime(2026, 11, 2, 15, 30, tzinfo=IST)
STAMP = "2026-11-02T10:00:00Z"
NAN = float("nan")

DECODING = {"runtime": "transformers"}
JUDGE = {
    "repo": "ToxicityPrompts/PolyGuard-Qwen",
    "revision": SHA,
    "dtype": "bfloat16",
    "decoding": {"runtime": "transformers", "max_new_tokens": 100},
    "prompt_format": "polyguard-v1",
    "parser_version": "1",
    "config_hash": DIGEST,
}
REF = {"kind": "model", "repo": "CohereLabs/tiny-aya-global", "revision": SHA, "sha256": None}
RATE = {
    "row": "en",
    "checkpoint": f"vincentnkq/tiny-aya-global-drift-merged@{SHA}",
    "metric": "safe",
    "k": 90,
    "n": 100,
    "rate": 0.9,
    "ci": [0.8256, 0.9448],
    "judge": JUDGE,
    "preamble": "keep",
}
DELTA = {
    "row": "pooled",
    "a": "baseline",
    "b": "drifted",
    "metric": "safe",
    "n_pairs": 600,
    "delta": -0.07,
    "ci": [-0.11, -0.03],
    "p_value": 0.001,
}

SAMPLES = {
    "run": {
        "run_id": "generate-baseline-1",
        "job_id": "6906f1c2",
        "kind": "generate",
        "mode": "full",
        "config_hash": DIGEST,
        "code_sha": SHA,
        "inputs": [REF],
        "outputs": [{**REF, "kind": "dataset", "repo": "vincentnkq/safety-drift-map-responses"}],
        "merged_sha256": None,
        "command": ["bash", "-lc", "python -m driftmap.scripts.generate"],
        "started": LOCAL,
        "finished": None,
        "wall_clock_s": None,
        "flavor": "l4x1",
        "price_micro_usd_per_min": 13_333,
        "minutes": None,
        "cost_micro_usd": None,
        "prompts": 1450,
        "prompt_tokens": 530_000,
        "generated_tokens": 120_000,
        "tokens_per_s": None,
        "status": "partial",
        "partial": {"resume_from": "harmful/hi", "done": ["harmful/en"], "written_at": LOCAL, "config_hash": DIGEST},
        "oom": None,
        "peak_vram_gb": 9.5,
        "peak_rss_gb": None,
        "install_minutes": 3.5,
        "notes": [],
    },
    "decision": {"decision": "proceed", "note": "drift is 7 points", "ts": LOCAL},
    "ledger": {
        "job_id": "6906f1c2",
        "run_id": "generate-baseline-1",
        "kind": "generate",
        "flavor": "l4x1",
        "mode": "full",
        "price_micro_usd_per_min": 13_333,
        "minutes": 41,
        "cost_micro_usd": 546_653,
        "stage": "COMPLETED",
        "record_status": "ok",
        "exit_code": 0,
        "billing_used_after_micro_usd": None,
        "timeout_s": 5400,
        "pending": False,
        "lost": False,
        "oom": False,
        "notes": [],
        "ts": LOCAL,
    },
    "plan": {
        "kind": "drift",
        "mode": "full",
        "runner": "pip",
        "script": "driftmap.scripts.drift",
        "flavor": "l4x1",
        "image": "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime",
        "timeout_s": 10_800,
        "estimate_minutes": 90,
        "price_micro_usd_per_min": 13_333,
        "cost_at_timeout_micro_usd": 2_399_940,
        "dependencies": ["safetune==0.1.10"],
        "env": {"PIP_BREAK_SYSTEM_PACKAGES": "1"},
        "secret_names": ["HF_TOKEN"],
        "code_sha": SHA,
        "script_args": ["--run-id", "drift-1"],
        "config_hash": DIGEST,
    },
    "generation": {
        "checkpoint": f"CohereLabs/tiny-aya-global@{SHA}",
        "language": "hi",
        "set": "harmful",
        "prompt_id": DIGEST,
        "prompt": "p",
        "rendered_prompt": "<BOS_TOKEN>p",
        "rendered_prefix_sha256": DIGEST,
        "response": None,
        "decoding": DECODING,
        "status": "unscored",
        "error": "empty",
        "prompt_tokens": 366,
        "generated_tokens": 0,
        "stop_reason": None,
    },
    "label": {
        "checkpoint": f"CohereLabs/tiny-aya-global@{SHA}",
        "language": "hi",
        "set": "harmful",
        "prompt_id": DIGEST,
        "prompt": "p",
        "rendered_prefix_sha256": DIGEST,
        "request_harm": "yes",
        "refusal": "yes",
        "response_harm": "no",
        "raw_output_present": True,
        "judge": JUDGE,
        "chrf": None,
    },
    "alpha_screen": {
        "alphas": [
            {
                "alpha": 0.25,
                "safe_rate_pooled": 0.9,
                "n_judged": 120,
                "chrf_pooled": 0.41,
                "pass_safety": True,
                "pass_chrf": True,
                "error": None,
            }
        ],
        "chosen": 0.25,
        "recover_status": "chosen",
        "badge": "Faithful",
        "safetune_version": "0.1.10",
        "dare": {},
        "judge": JUDGE,
        "bars": {},
        "baseline_safe_pooled": 0.93,
        "drifted_chrf_subset": 0.42,
        "tied_weight_owner": "model.embed_tokens.weight",
    },
    "gguf": {
        "llamacpp_tag": "b11960",
        "tarball": "llama-b11960-bin-ubuntu-x64.tar.gz",
        "tarball_sha256": DIGEST,
        "quant": "q4_k_m",
        "sha256": DIGEST,
        "bytes": 2_140_000_000,
        "source_repo": "vincentnkq/tiny-aya-global-repaired",
        "source_revision": SHA,
    },
    "manifest": {
        "seed": 20261102,
        "seed_derivation": "sha256(f'{seed}:{language}')[:8] big-endian",
        "per_language": {"en": {"harmful": 100, "undersampled": False, "benign_train": 423, "benign_heldout": 100}},
        "sources": [{**REF, "kind": "dataset", "repo": "CohereLabs/aya_redteaming"}],
        "licences": {"CohereLabs/aya_redteaming": "apache-2.0"},
        "dropped_by_dedup": 3,
        "dropped_over_length": 0,
        "dropped_special_token_text": 0,
        "max_rendered_tokens": {"harmful": 512},
        "rendered_prefix_sha256": DIGEST,
    },
    "tables": {
        "columns": {"baseline": {"state": "tabled", "reason": None, "kind": "baseline"}},
        "safety": [RATE],
        "over_refusal": [{**RATE, "metric": "over_refusal"}],
        "chrf": [{"row": "en", "checkpoint": RATE["checkpoint"], "n": 100, "chrf": 0.41}],
        "deltas": [DELTA],
        "flags": {"drift_below_threshold": False},
        "judge": JUDGE,
        "seeds": {"seed": 20261102, "bootstrap": 42},
    },
}
MODELS = {
    "run": RunRecord,
    "decision": records.Decision,
    "ledger": LedgerLine,
    "plan": records.Plan,
    "generation": records.GenerationRow,
    "label": records.LabelRow,
    "alpha_screen": records.AlphaScreen,
    "gguf": records.GgufRecord,
    "manifest": records.Manifest,
    "tables": Tables,
}
SNAPSHOT = {
    "billing_used_micro_usd": None,
    "ledger_total_micro_usd": 546_653,
    "smoke_spent_micro_usd": 0,
    "smoke_pending": False,
    "prices_micro_usd_per_min": {"l4x1": 13_333},
    "ceilings": {},
    "in_flight_gpu": [],
    "smoke_records": [
        {
            "script": "driftmap.scripts.drift",
            "image": "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime",
            "runner": "pip",
            "dependencies": ["safetune==0.1.10"],
            "schema_versions": schema_versions(),
            "status": "ok",
            "bos_check": None,
            "install_minutes": 3.5,
            "upload_mb_s": 11.0,
            "peak_vram_gb": 9.5,
            "peak_rss_gb": 6.0,
            "tokens_per_s": 900.0,
        }
    ],
    "gates": {"CohereLabs/tiny-aya-base": "403"},
    "lost_runs": [["drift", 10_800]],
    "oom_runs": [["generate", DIGEST]],
    "table_on_file": False,
    "drift_flag": None,
    "drift_decision": None,
    "code_on_origin": True,
}


def _payload(name, /, **changes):
    return {"schema": schema_versions()[name], **SAMPLES[name], **changes}


def _nulls(node, path=""):
    if node is None:
        yield path
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _nulls(value, f"{path}.{key}")
    elif isinstance(node, list):
        for value in node:
            yield from _nulls(value, path)


def _locs(caught):
    return [error["loc"] for error in caught.value.errors()]


@pytest.fixture
def kolkata(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_every_record_round_trips_and_rejects_bad_input(tmp_path, kolkata):
    """Records keep "schema" and UTC stamps on disk and refuse malformed input (TS-CFG-5)."""
    assert time.localtime().tm_gmtoff == 19_800
    assert MODELS.keys() == schema_versions().keys()

    for kind, cls in MODELS.items():
        model = cls.model_validate(_payload(kind))
        path = tmp_path / kind / "record.json"
        dump(model, path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema"] == schema_versions()[kind], kind
        assert "schema_" not in raw, kind
        assert load(cls, path) == model, kind
        assert cls.model_validate(model.model_dump()) == model, kind
        assert json.loads(model.model_dump_json())["schema"] == raw["schema"], kind
        assert [p.name for p in path.parent.iterdir()] == ["record.json"], kind

        with pytest.raises(ValidationError) as caught:
            cls.model_validate(_payload(kind, schema="driftmap.other/1"))
        assert _locs(caught) == [("schema",)], kind
        assert caught.value.errors()[0]["type"] == "literal_error", kind

        with pytest.raises(ValidationError) as caught:
            cls.model_validate(_payload(kind, surprise=1))
        assert _locs(caught) == [("surprise",)], kind
        assert caught.value.errors()[0]["type"] == "extra_forbidden", kind

        with pytest.raises(ValidationError) as caught:
            cls.model_validate(SAMPLES[kind])
        assert _locs(caught) == [("schema",)], kind

    run = json.loads((tmp_path / "run" / "record.json").read_text(encoding="utf-8"))
    ledger = json.loads((tmp_path / "ledger" / "record.json").read_text(encoding="utf-8"))
    decision = json.loads((tmp_path / "decision" / "record.json").read_text(encoding="utf-8"))
    assert run["started"] == run["partial"]["written_at"] == ledger["ts"] == decision["ts"] == STAMP
    loaded = load(RunRecord, tmp_path / "run" / "record.json")
    assert loaded.started == LOCAL
    assert loaded.started.utcoffset() == timedelta(0)

    for kind, field in [("run", "started"), ("ledger", "ts"), ("decision", "ts")]:
        with pytest.raises(ValidationError) as caught:
            MODELS[kind].model_validate(_payload(kind, **{field: datetime(2026, 11, 2, 10, 0)}))
        assert _locs(caught) == [(field,)], kind
        assert caught.value.errors()[0]["type"] == "timezone_aware", kind

    with pytest.raises(ValidationError) as caught:
        RateRow.model_validate({**RATE, "rate": NAN})
    assert _locs(caught) == [("rate",)]
    assert caught.value.errors()[0]["type"] == "finite_number"

    with pytest.raises(ValidationError) as caught:
        LedgerLine.model_validate(_payload("ledger", cost_micro_usd="5.00"))
    assert _locs(caught) == [("cost_micro_usd",)]

    row = RateRow.model_validate(RATE)
    with pytest.raises(ValidationError):
        row.rate = NAN
    assert "null" not in row.model_dump_json()


def _money_fields():
    for _, cls in inspect.getmembers(records, inspect.isclass):
        if not issubclass(cls, BaseModel) or cls.__module__ != records.__name__:
            continue
        for name, field in cls.model_fields.items():
            if "micro_usd" in name or cls is Ceilings or (cls is Refusal and name in ("estimate", "ceiling")):
                yield cls.__name__, name, field.rebuild_annotation()


def test_an_alpha_screen_can_wait_for_its_judge():
    unjudged = {"safe_rate_pooled": None, "n_judged": 0, "pass_safety": None}
    row = SAMPLES["alpha_screen"]["alphas"][0] | unjudged
    waiting = {
        "alphas": [row],
        "chosen": None,
        "recover_status": "screened",
        "judge": None,
        "baseline_safe_pooled": None,
    }

    screen = records.AlphaScreen.model_validate(_payload("alpha_screen", **waiting))

    assert (screen.recover_status, screen.judge, screen.baseline_safe_pooled) == ("screened", None, None)
    # rows written before the counts existed read as zero; new ones keep theirs
    assert (screen.alphas[0].n_benign_scored, screen.alphas[0].n_benign_empty) == (0, 0)
    figures = {"n_benign_scored": 78, "n_benign_empty": 42, "n_harmful": 120, "rows_sha256": DIGEST}
    counted = records.AlphaRow.model_validate(row | figures)
    assert records.AlphaRow.model_validate_json(counted.model_dump_json()) == counted
    assert counted.model_dump(include=set(figures)) == figures
    assert (screen.alphas[0].n_harmful, screen.alphas[0].rows_sha256, screen.undecided) == (0, None, None)
    with pytest.raises(ValidationError):
        records.AlphaRow.model_validate(row | {"rows_sha256": "abc"})
    open_ = records.AlphaScreen.model_validate(_payload("alpha_screen", **waiting, undecided="too few verdicts"))
    assert open_.undecided == "too few verdicts"
    assert records.AlphaScreen.model_validate_json(screen.model_dump_json(by_alias=True)) == screen
    for field in ("judge", "baseline_safe_pooled"):
        payload = _payload("alpha_screen")
        del payload[field]
        with pytest.raises(ValidationError) as caught:
            records.AlphaScreen.model_validate(payload)
        assert _locs(caught) == [(field,)]
    with pytest.raises(ValidationError) as caught:
        records.AlphaScreen.model_validate(_payload("alpha_screen", recover_status="pending"))
    assert _locs(caught) == [("recover_status",)]


def test_floats_are_finite_and_money_is_a_strict_int():
    """NaN, float money and string money are refused; tables hold no NaN (TS-CFG-13)."""
    with pytest.raises(ValidationError) as caught:
        RateRow.model_validate({**RATE, "rate": NAN})
    assert _locs(caught) == [("rate",)]

    with pytest.raises(ValidationError) as caught:
        DeltaRow.model_validate({**DELTA, "ci": (NAN, NAN)})
    assert _locs(caught) == [("ci", 0), ("ci", 1)]

    for bad in (5.0, "5000000", True):
        with pytest.raises(ValidationError) as caught:
            RunRecord.model_validate(_payload("run", cost_micro_usd=bad))
        assert _locs(caught)[0][0] == "cost_micro_usd", bad

    fields = list(_money_fields())
    assert {"RunRecord", "LedgerLine", "Plan", "Snapshot", "Ceilings", "Refusal"} == {owner for owner, _, _ in fields}
    for owner, name, annotation in fields:
        adapter = TypeAdapter(annotation)
        wrap = (lambda value: {"l4x1": value}) if name == "prices_micro_usd_per_min" else (lambda value: value)
        assert adapter.validate_python(wrap(5_000_000)) == wrap(5_000_000), (owner, name)
        for bad in (5.0, "5000000", "5.00"):
            with pytest.raises(ValidationError):
                adapter.validate_python(wrap(bad))

    cell = {**DELTA, "row": "hi", "n_pairs": 1, "delta": None, "ci": None, "p_value": None}
    empty = {**RATE, "row": "ar", "k": 0, "n": 0, "rate": None, "ci": None}
    tables = Tables.model_validate(_payload("tables", deltas=[DELTA, cell], safety=[RATE, empty]))
    text = tables.model_dump_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert set(_nulls(json.loads(text))) == {
        ".columns.baseline.reason",
        ".deltas.delta",
        ".deltas.ci",
        ".deltas.p_value",
        ".safety.rate",
        ".safety.ci",
    }


def test_revision_must_be_a_commit_sha():
    """A branch name is not a revision (TS-NFR-3)."""
    for kind in ("model", "dataset", "code"):
        assert Ref.model_validate({**REF, "kind": kind}).revision == SHA
        assert Ref.model_validate({**REF, "kind": kind, "sha256": DIGEST}).sha256 == DIGEST
        for bad in ("main", SHA.upper(), SHA[:39], SHA + "0", SHA + "\n", "refs/pr/1", ""):
            with pytest.raises(ValidationError) as caught:
                Ref.model_validate({**REF, "kind": kind, "revision": bad})
            assert _locs(caught) == [("revision",)], (kind, bad)

    with pytest.raises(ValidationError) as caught:
        RunRecord.model_validate(_payload("run", inputs=[{**REF, "revision": "main"}]))
    assert _locs(caught) == [("inputs", 0, "revision")]

    for kind, tag in (("binary", "b11960"), ("image", "2.14.1-cuda12.6-cudnn9-runtime")):
        pinned = {"kind": kind, "repo": "ggml-org/llama.cpp", "revision": tag, "sha256": DIGEST}
        assert Ref.model_validate(pinned).revision == tag
        with pytest.raises(ValidationError) as caught:
            Ref.model_validate({**pinned, "revision": ""})
        assert _locs(caught) == [("revision",)], kind
        for missing in ({**pinned, "sha256": None}, {k: v for k, v in pinned.items() if k != "sha256"}):
            with pytest.raises(ValidationError) as caught:
                Ref.model_validate(missing)
            assert _locs(caught) == [("sha256",)], kind
        for bad in (DIGEST[:63], DIGEST.upper(), "sha256:" + DIGEST, SHA):
            with pytest.raises(ValidationError) as caught:
                Ref.model_validate({**pinned, "sha256": bad})
            assert _locs(caught) == [("sha256",)], (kind, bad)

    with pytest.raises(ValidationError) as caught:
        Ref.model_validate({**REF, "sha256": "not-a-digest"})
    assert _locs(caught) == [("sha256",)]

    for cls, name in ((RunRecord, "run"), (records.Plan, "plan")):
        for bad in ("main", "HEAD", SHA[:12], SHA + "'; id #"):
            with pytest.raises(ValidationError) as caught:
                cls.model_validate(_payload(name, code_sha=bad))
            assert _locs(caught) == [("code_sha",)], (name, bad)


def test_ceilings_defaults():
    """Ceilings default to the agreed micro-USD amounts (TS-NFR-4)."""
    defaults = Ceilings().model_dump()
    assert defaults == {
        "per_run": 5_000_000,
        "smoke": 2_000_000,
        "soft": 16_000_000,
        "hard": 20_000_000,
        "own_money": 0,
        "envelope": 13_600_000,
    }
    assert list(defaults) == ["per_run", "smoke", "soft", "hard", "own_money", "envelope"]
    assert all(type(value) is int for value in defaults.values())


@pytest.mark.parametrize(
    ("changes", "loc"),
    [
        ({"smoke": 5_000_001}, ()),
        ({"per_run": 16_000_001}, ()),
        ({"soft": 20_000_001}, ()),
        ({"hard": 15_999_999}, ()),
        ({"smoke": -1}, ("smoke",)),
        ({"hard": -1}, ("hard",)),
        ({"own_money": -1}, ("own_money",)),
        ({"envelope": -1}, ("envelope",)),
    ],
)
def test_ceilings_are_ordered_and_never_negative(changes, loc):
    with pytest.raises(ValidationError) as caught:
        Ceilings(**changes)
    assert _locs(caught) == [loc]

    ceilings = Ceilings()
    ((name, value),) = changes.items()
    with pytest.raises(ValidationError):
        setattr(ceilings, name, value)
    assert ceilings == Ceilings()


def test_ceilings_may_be_equal():
    flat = Ceilings(smoke=3_000_000, per_run=3_000_000, soft=3_000_000, hard=3_000_000)
    assert flat.smoke == flat.hard == 3_000_000
    assert Ceilings(smoke=0, per_run=0, soft=0, hard=0, envelope=0).hard == 0


def test_snapshot_round_trips_with_tuples_and_default_ceilings(tmp_path):
    snapshot = Snapshot.model_validate(SNAPSHOT)
    assert snapshot.ceilings == Ceilings()
    assert snapshot.lost_runs == [("drift", 10_800)]
    assert snapshot.oom_runs == [("generate", DIGEST)]
    dump(snapshot, tmp_path / "snapshot.json")
    assert load(Snapshot, tmp_path / "snapshot.json") == snapshot

    with pytest.raises(ValidationError) as caught:
        Snapshot.model_validate({**SNAPSHOT, "gates": {"CohereLabs/tiny-aya-base": "500"}})
    assert _locs(caught) == [("gates", "CohereLabs/tiny-aya-base")]


def test_ledger_stage_takes_the_library_member_or_its_name():
    by_name = LedgerLine.model_validate(_payload("ledger"))
    by_member = LedgerLine.model_validate(_payload("ledger", stage=JobStage.COMPLETED))
    assert by_name == by_member
    assert by_name.stage is JobStage.COMPLETED
    assert json.loads(by_member.model_dump_json())["stage"] == "COMPLETED"


def test_ledger_line_keeps_a_stage_the_library_does_not_know(tmp_path):
    line = LedgerLine.model_validate(_payload("ledger", stage="UPDATING"))
    assert line.stage == "UPDATING"
    assert type(line.stage) is str
    dump(line, tmp_path / "line.json")
    assert json.loads((tmp_path / "line.json").read_text(encoding="utf-8"))["stage"] == "UPDATING"
    assert load(LedgerLine, tmp_path / "line.json") == line
    for bad in ("", 5, None):
        with pytest.raises(ValidationError):
            LedgerLine.model_validate(_payload("ledger", stage=bad))


def test_ledger_line_config_hash_is_optional(tmp_path):
    assert LedgerLine.model_validate(_payload("ledger")).config_hash is None
    line = LedgerLine.model_validate(_payload("ledger", config_hash=DIGEST))
    dump(line, tmp_path / "line.json")
    assert json.loads((tmp_path / "line.json").read_text(encoding="utf-8"))["config_hash"] == DIGEST
    assert load(LedgerLine, tmp_path / "line.json") == line
    with pytest.raises(ValidationError):
        LedgerLine.model_validate(_payload("ledger", config_hash=5))


def test_ledger_and_plan_amounts_are_never_negative():
    for field in ("cost_micro_usd", "minutes"):
        with pytest.raises(ValidationError) as caught:
            LedgerLine.model_validate(_payload("ledger", **{field: -1}))
        assert _locs(caught) == [(field,)]
        assert getattr(LedgerLine.model_validate(_payload("ledger", **{field: 0})), field) == 0
    with pytest.raises(ValidationError) as caught:
        records.Plan.model_validate(_payload("plan", cost_at_timeout_micro_usd=-1))
    assert _locs(caught) == [("cost_at_timeout_micro_usd",)]


def test_job_kinds_and_labels():
    assert records.JobKind.__args__ == ("drift", "generate", "judge", "repair", "steer", "quant", "quant_eval")
    assert records.Lbl.__args__ == ("yes", "no", "unscored", "n/a", "judge_unsupported")
    with pytest.raises(ValidationError):
        LedgerLine.model_validate(_payload("ledger", kind="train"))


def test_run_record_smoke_measurements_default_to_none():
    plain = RunRecord.model_validate(_payload("run"))
    assert (plain.bos_check, plain.upload_mb_s) == (None, None)
    smoke = RunRecord.model_validate(_payload("run", bos_check=True, upload_mb_s=11.0))
    assert (smoke.bos_check, smoke.upload_mb_s) == (True, 11.0)
    with pytest.raises(ValidationError):
        RunRecord.model_validate(_payload("run", bos_check="maybe"))


def test_refusal_and_plan_defaults():
    refusal = Refusal(guard="per_run_ceiling", reason="5.01 > 5.00", estimate=5_013_208, ceiling=5_000_000)
    assert Refusal(guard="gate", reason="403").estimate is None
    assert refusal.model_dump()["ceiling"] == 5_000_000
    plan = records.Plan.model_validate(_payload("plan"))
    assert (plan.override_smoke, plan.override_soft) == (False, False)
    assert records.Manifest.model_validate(_payload("manifest")).n_harmful == 600
    assert records.Manifest.model_validate(_payload("manifest")).note == ""
    assert records.Manifest.model_validate(_payload("manifest", note="skipped languages: sr 3")).note.endswith("sr 3")


def test_dump_replaces_the_file_whole(tmp_path):
    path = tmp_path / "decisions" / "drift.json"
    dump(records.Decision.model_validate(_payload("decision")), path)
    dump(records.Decision.model_validate(_payload("decision", decision="extend")), path)
    assert load(records.Decision, path).decision == "extend"
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert [p.name for p in path.parent.iterdir()] == ["drift.json"]


def test_dump_keeps_the_old_file_when_the_swap_fails(tmp_path, monkeypatch):
    path = tmp_path / "decisions" / "drift.json"
    dump(records.Decision.model_validate(_payload("decision")), path)

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        dump(records.Decision.model_validate(_payload("decision", decision="extend")), path)
    assert load(records.Decision, path).decision == "proceed"
    assert [p.name for p in path.parent.iterdir()] == ["drift.json"]


def test_only_an_api_runtime_leaves_greedy_decoding_unclaimed():
    assert Decoding(runtime="transformers").greedy is True and Decoding(runtime="llama-server").greedy is True
    assert Decoding(runtime="api", greedy=False).model_dump()["greedy"] is False
    for fields in ({"runtime": "transformers", "greedy": False}, {"runtime": "api"}, {"runtime": "batch"}):
        with pytest.raises(ValidationError):
            Decoding(**fields)
    closed = JudgeConfig(**JUDGE | {"prompt_format": "rubric-v1", "decoding": {"runtime": "api", "greedy": False}})
    assert closed.prompt_format == "rubric-v1"
    with pytest.raises(ValidationError):
        JudgeConfig(**JUDGE | {"prompt_format": "rubric-v2"})
