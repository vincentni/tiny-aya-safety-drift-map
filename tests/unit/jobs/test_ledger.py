import fcntl
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from huggingface_hub import JobStage
from hypothesis import given
from hypothesis import strategies as st

from driftmap.core.records import LedgerLine
from driftmap.core.schemas import ExitCode
from driftmap.core.settings import load_settings
from driftmap.jobs import ledger as ledger_module
from driftmap.jobs.ledger import Ledger, LedgerCorrupt, PendingAfterTerminal, pending_line, terminal_line
from driftmap.jobs.plan import build_plan
from tests.fakes.settings import settings_toml

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "ledger"
REPO_ROOT = Path(__file__).resolve().parents[3]
SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 5, 18, 15, tzinfo=UTC)
LATER = NOW + timedelta(minutes=40)
DRIFT, GENERATE, SMOKE, FALLBACK = (
    "69076d2e8f1a4c0b9e3d5a71",
    "6907a41b3c9e2f7d0a6b8c15",
    "6906e0c47a2d9b1f3e5c8a02",
    "6908b2f95d1c7e3a4b0f6d29",
)


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "results" / "ledger.jsonl")


def _plan(settings, kind="generate", mode="full", **kwargs):
    return build_plan(settings, kind, mode, None, SHA, now=NOW, repo_root=REPO_ROOT, **kwargs)


def _write(ledger, rows):
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def _job(row):
    return LedgerLine.model_validate_json(row).job_id


@st.composite
def _interleavings(draw, rows):
    """The rows shuffled across jobs, each job's own rows kept in order."""
    queues = {}
    for row in rows:
        queues.setdefault(_job(row), []).append(row)
    slots = draw(st.permutations([_job(row) for row in rows]))
    taken = dict.fromkeys(queues, 0)
    ordered = []
    for job in slots:
        ordered.append(queues[job][taken[job]])
        taken[job] += 1
    return ordered


def test_totals_take_the_last_line_of_each_job(ledger):
    """Totals count each job's last line under any interleaving; a float amount is corrupt (TS-LCH-5)."""
    rows = (FIXTURES / "pending_terminal.jsonl").read_text(encoding="utf-8").splitlines()
    assert [LedgerLine.model_validate_json(row).cost_micro_usd for row in rows] == [
        399_990,
        159_996,
        2_399_940,
        1_199_970,
        639_984,
        1_500_030,
        1_000_020,
    ]

    @given(order=_interleavings(rows))
    def totals_hold(order):
        _write(ledger, order)
        assert ledger.total_micro_usd() == 2_999_970
        assert ledger.smoke_spent_micro_usd() == 159_996
        assert type(ledger.total_micro_usd()) is int
        last = ledger.read()
        assert set(last) == {DRIFT, GENERATE, SMOKE, FALLBACK}
        assert (last[DRIFT].pending, last[GENERATE].pending) == (False, True)

    totals_hold()

    _write(ledger, [*rows[:4], rows[4].replace('"cost_micro_usd":639984', '"cost_micro_usd":639984.0'), *rows[5:]])
    with pytest.raises(LedgerCorrupt, match=r"line 5: cost_micro_usd"):
        ledger.total_micro_usd()


def test_torn_line_stops_every_reader(ledger):
    """A cut-off line, last or not, raises with the file and line number (TS-LCH-6)."""
    torn = (FIXTURES / "torn.jsonl").read_text(encoding="utf-8")
    whole = (FIXTURES / "pending_terminal.jsonl").read_text(encoding="utf-8").splitlines()
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_text(torn, encoding="utf-8")
    readers = (
        ledger.read,
        ledger.total_micro_usd,
        ledger.smoke_spent_micro_usd,
        ledger.smoke_pending,
        ledger.lost_runs,
        ledger.oom_runs,
    )
    for reader in readers:
        with pytest.raises(LedgerCorrupt, match=r"ledger\.jsonl: line 3") as caught:
            reader()
        assert caught.value.exit_code is ExitCode.UNEXPECTED == 1
    with pytest.raises(LedgerCorrupt, match="line 3"):
        ledger.append(LedgerLine.model_validate_json(whole[4]))
    assert ledger.path.read_text(encoding="utf-8") == torn

    _write(ledger, [whole[2], whole[3][:90], whole[4]])
    for reader in readers:
        with pytest.raises(LedgerCorrupt, match=r"ledger\.jsonl: line 2"):
            reader()
    _write(ledger, [whole[2], "", whole[4]])
    with pytest.raises(LedgerCorrupt, match="line 2"):
        ledger.read()


@pytest.mark.parametrize(
    ("running_secs", "minutes", "cost"),
    [(0, 0, 0), (59, 1, 13_333), (60, 1, 13_333), (61, 2, 26_666)],
)
def test_terminal_line_bills_started_minutes(settings, ledger, running_secs, minutes, cost):
    """A pending line stands at the timeout cost until the terminal line bills started minutes (TS-LCH-16)."""
    pending = pending_line(_plan(settings), GENERATE, NOW, "settings")
    ledger.append(pending)
    before = ledger.path.read_bytes()
    assert (pending.pending, pending.minutes, pending.cost_micro_usd) == (True, 90, 1_199_970)
    assert (pending.timeout_s, pending.run_id, pending.stage) == (5400, "generate-full-20261105T181500Z", "SCHEDULING")
    assert pending.config_hash == settings.config_hash()
    assert ledger.total_micro_usd() == 1_199_970

    done = terminal_line(pending, running_secs=running_secs, stage="COMPLETED", ts=LATER, record_status="ok")
    ledger.append(done)
    assert ledger.path.read_bytes().startswith(before)
    assert ledger.path.read_bytes().count(b"\n") == 2
    assert (done.pending, done.minutes, done.cost_micro_usd) == (False, minutes, cost)
    assert type(done.cost_micro_usd) is int
    assert (done.job_id, done.run_id, done.kind, done.timeout_s) == (GENERATE, pending.run_id, "generate", 5400)
    assert done.config_hash == pending.config_hash
    assert ledger.total_micro_usd() == cost
    assert ledger.read()[GENERATE] == done


def test_unknown_running_time_keeps_the_timeout_cost(settings):
    pending = pending_line(_plan(settings), GENERATE, NOW, "hardware")
    assert pending.notes == ["price=hardware"]
    for stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED", "RUNNING", "UPDATING", JobStage.ERROR):
        done = terminal_line(pending, running_secs=None, stage=stage, ts=LATER)
        assert (done.pending, done.minutes, done.cost_micro_usd) == (False, 90, 1_199_970), stage
        assert done.notes == ["price=hardware", "running-time-unknown"]
    for stage in ("SCHEDULING", JobStage.SCHEDULING):
        never = terminal_line(pending, running_secs=None, stage=stage, ts=LATER)
        assert (never.pending, never.minutes, never.cost_micro_usd, never.notes) == (False, 0, 0, ["price=hardware"])
    instant = terminal_line(pending, running_secs=0, stage="ERROR", ts=LATER)
    assert (instant.cost_micro_usd, instant.notes) == (0, ["price=hardware"])
    for negative in (-1, -60, -3_600):
        with pytest.raises(ValueError, match="running_secs"):
            terminal_line(pending, running_secs=negative, stage="ERROR", ts=LATER)


def test_one_lock_covers_each_read_and_each_append(settings, ledger, monkeypatch):
    taken = []
    real = fcntl.flock

    def recording(handle, operation):
        taken.append(operation)
        return real(handle, operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", recording)
    pending = pending_line(_plan(settings), GENERATE, NOW, "settings")
    assert ledger.lines() == []
    assert taken == []
    ledger.append(pending)
    assert taken == [fcntl.LOCK_EX]
    assert ledger.total_micro_usd() == 1_199_970
    assert taken == [fcntl.LOCK_EX, fcntl.LOCK_SH]
    monkeypatch.undo()

    done = terminal_line(pending, running_secs=600, stage="COMPLETED", ts=LATER)
    writer = threading.Thread(target=ledger.append, args=(done,))
    with ledger.path.open("rb") as reader:
        fcntl.flock(reader, fcntl.LOCK_SH)
        writer.start()
        writer.join(timeout=0.3)
        assert writer.is_alive()
        assert len(ledger.lines()) == 1
    writer.join(timeout=30)
    assert not writer.is_alive()
    assert ledger.read()[GENERATE] == done


def test_pending_never_follows_a_terminal_line(settings, ledger):
    """A terminal-only job counts once, a late pending line is refused, a later line wins (TS-EDGE-3)."""
    pending = pending_line(_plan(settings), GENERATE, NOW, "settings")
    done = terminal_line(pending, running_secs=2_400, stage="COMPLETED", ts=LATER)
    ledger.append(done)
    assert ledger.total_micro_usd() == 533_320

    before = ledger.path.read_bytes()
    with pytest.raises(PendingAfterTerminal, match=GENERATE):
        ledger.append(pending)
    assert ledger.path.read_bytes() == before

    reconciled = done.model_copy(update={"minutes": 41, "cost_micro_usd": 546_653, "notes": [*done.notes, "reconcile"]})
    ledger.append(reconciled)
    assert ledger.total_micro_usd() == 546_653
    assert len(ledger.lines()) == 2


def test_views_come_from_last_lines(settings, ledger):
    assert (ledger.read(), ledger.total_micro_usd(), ledger.smoke_pending()) == ({}, 0, False)
    assert (ledger.lost_runs(), ledger.oom_runs()) == ([], [])

    generate, repair = _plan(settings), _plan(settings, "repair", override_smoke=True, override_soft=True)
    smoke = _plan(settings, "judge", "smoke")
    lost = pending_line(generate, GENERATE, NOW, "settings")
    oom = pending_line(repair, DRIFT, NOW, "settings")
    probe = pending_line(smoke, SMOKE, NOW, "settings")
    for row in (lost, oom, probe):
        ledger.append(row)
    assert ledger.smoke_pending() is True
    assert ledger.smoke_spent_micro_usd() == 399_990
    assert (ledger.lost_runs(), ledger.oom_runs()) == ([], [])
    assert oom.config_hash == settings.config_hash()
    assert oom.notes == ["price=settings", "override-smoke", "override-soft"]
    assert (lost.config_hash, lost.notes) == (settings.config_hash(), ["price=settings"])

    ledger.append(terminal_line(lost, running_secs=5_400, stage="ERROR", ts=LATER, lost=True))
    killed = terminal_line(oom, running_secs=900, stage="ERROR", ts=LATER, oom=True)
    assert killed.config_hash == settings.config_hash()
    assert killed.notes == ["price=settings", "override-smoke", "override-soft"]
    again = ["override-soft", "sigkill-suspected"]
    suspected = terminal_line(oom, running_secs=900, stage="ERROR", ts=LATER, notes=again)
    assert suspected.notes == ["price=settings", "override-smoke", "override-soft", "sigkill-suspected"]
    # a terminal line built elsewhere without the hash
    ledger.append(killed.model_copy(update={"config_hash": None}))
    ledger.append(terminal_line(probe, running_secs=540, stage="COMPLETED", ts=LATER, notes=["partial"]))
    assert ledger.lost_runs() == [("generate", 5400)]
    assert ledger.oom_runs() == [("repair", settings.config_hash())]
    assert ledger.smoke_pending() is False
    assert ledger.smoke_spent_micro_usd() == 119_997
    assert ledger.read()[SMOKE].notes == ["price=settings", "partial"]
    assert ledger.read()[DRIFT].config_hash is None
    assert ledger.total_micro_usd() == 1_199_970 + 199_995 + 119_997

    # a later line without the flag does not clear it
    for job in (GENERATE, DRIFT):
        ledger.append(ledger.read()[job].model_copy(update={"lost": False, "oom": False, "notes": ["reconcile"]}))
    assert ledger.lost_runs() == [("generate", 5400)]
    assert ledger.oom_runs() == [("repair", settings.config_hash())]
