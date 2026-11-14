import ast
import functools
import json
import logging
import shutil
import string
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils._http import default_client_factory
from hypothesis import HealthCheck, given
from hypothesis import settings as examples
from hypothesis import strategies as st
from typer.testing import CliRunner

from driftmap import cli
from driftmap.core import clients, ids
from driftmap.core import hub as core_hub
from driftmap.core.hub import Hub
from driftmap.core.records import (
    AlphaRow,
    AlphaScreen,
    Bars,
    Dare,
    Decision,
    Decoding,
    JudgeConfig,
    Ref,
    RunRecord,
    Seeds,
    SmokeRecord,
    Tables,
    dump,
)
from driftmap.core.schemas import (
    SCHEMA_ALPHA_SCREEN,
    SCHEMA_RUN,
    SCHEMA_TABLES,
    DriftmapError,
    ExitCode,
    schema_versions,
)
from driftmap.core.settings import load_settings
from driftmap.data import build as data_build
from driftmap.jobs import launcher, readiness
from driftmap.jobs.client import HfJobsClient
from driftmap.jobs.kinds import State
from driftmap.jobs.ledger import Ledger
from driftmap.jobs.plan import build_plan, pins
from driftmap.jobs.readiness import ITEMS, READY_PATH, Check, Item, ReadyReport
from driftmap.jobs.snapshot import DECISION_PATH, LOCAL, SMOKE_DIR, TABLES_PATH, Deps
from driftmap.judge import closed, reparse
from driftmap.table import build as table_build
from driftmap.table import render
from driftmap.table.build import NOT_COMPARABLE
from tests.fakes import FakeHub, FakeJobsClient
from tests.fakes import labels as fake_labels
from tests.fakes.datasets import (
    BENIGN,
    HARMFUL,
    SIX,
    XSTEST,
    benign_rows,
    fixture_rows,
    harmful_rows,
    heads,
    source_hub,
)
from tests.fakes.settings import settings_toml

REPO_ROOT = Path(__file__).resolve().parents[2]
AYA, BASE = "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"
OWNER = "vincentnkq"
RESULTS = f"{OWNER}/safety-drift-map-results"
EVAL = f"{OWNER}/aya-redteaming-6lang-eval"
SHA = "0123456789abcdef0123456789abcdef01234567"
TOKEN = "hf_" + "b" * 34
HOME_DIR = "/" + "home/someone"
VERBS = ["chain", "data", "decide", "judge", "launch", "ledger", "plan", "ready", "scan", "status", "table", "wait"]
SUBMIT = FakeJobsClient.run_job
HIDDEN = {"HF_TOKEN": "***"}
BASELINE = ["--", "--kind", "baseline"]
CHECKPOINT = Ref(kind="model", repo=AYA, revision="1" * 40)


def git(argv, **kwargs):
    """Every git call succeeds and HEAD is SHA."""
    return subprocess.CompletedProcess(argv, 0, f"{SHA}\n" if "rev-parse" in argv else "", "")


def run(*argv):
    return CliRunner().invoke(cli.app, list(argv))


def flat(result):
    """Output on one line, without the frame click draws around an error."""
    return " ".join(result.output.replace("│", " ").split())


def judge_config(settings):
    return JudgeConfig(
        repo=settings.judge.closed.name,
        revision="n/a",
        dtype="n/a",
        decoding=Decoding(greedy=False, runtime="api"),
        prompt_format="rubric-v1",
        parser_version="1",
        config_hash="h",
    )


def run_record(run_id, config_hash, status="ok", **changes):
    kind, mode, _ = run_id.split("-")
    fields = {
        "schema": SCHEMA_RUN,
        "run_id": run_id,
        "job_id": None,
        "kind": kind,
        "mode": mode,
        "config_hash": config_hash,
        "code_sha": SHA,
        "inputs": [CHECKPOINT, Ref(kind="dataset", repo=EVAL, revision="6" * 40)] if kind == "generate" else [],
        "outputs": [],
        "merged_sha256": None,
        "command": [],
        "started": datetime(2026, 11, 2, 10, 2, tzinfo=UTC),
        "finished": datetime(2026, 11, 2, 10, 27, tzinfo=UTC),
        "wall_clock_s": 1500.0,
        "flavor": "l4x1",
        "price_micro_usd_per_min": 13_333,
        "minutes": 25,
        "cost_micro_usd": 333_325,
        "prompts": 10,
        "prompt_tokens": 100,
        "generated_tokens": 100,
        "tokens_per_s": 80.0,
        "status": status,
        "partial": None,
        "oom": None,
        "peak_vram_gb": 9.4,
        "peak_rss_gb": 6.1,
        "install_minutes": 2.5,
        "notes": [],
    }
    return RunRecord.model_validate(fields | changes)


class World:
    """A checkout with a passed readiness report, a fake Hub and scripted git."""

    def __init__(self, monkeypatch, root, hub=None, **overrides):
        self.root = root
        monkeypatch.chdir(root)
        self.config = settings_toml(root, **overrides)
        shutil.copy(REPO_ROOT / "pyproject.toml", root)
        self.hub = hub or FakeHub(root.with_name(f"{root.name}-hub"))
        for repo, kind in ((AYA, {"gated": True}), (BASE, {"gated": True}), (RESULTS, {"repo_type": "dataset"})):
            if repo not in self.hub.repos:
                self.hub.add_repo(repo, **kind)
        self.hub.authorize(AYA)
        self.client_log = root.with_name(f"{root.name}-client.jsonl")
        self.ledger = Ledger(root / launcher.LEDGER_PATH)
        self.secrets = []
        monkeypatch.setenv("DRIFTMAP_CLIENT", f"fake:{self.client_log}")
        # the attribute the real client keeps its HfApi under
        monkeypatch.setattr(FakeJobsClient, "api", self.hub, raising=False)
        monkeypatch.setattr(FakeJobsClient, "run_job", lambda client, **kwargs: self.submitted(client, **kwargs))
        monkeypatch.setattr(cli, "Deps", functools.partial(Deps, run=git))
        self.ready()

    @property
    def settings(self):
        return load_settings(self.config)

    def ready(self, **fields):
        items = [Item(number=number, name=name, status="PASS", detail="ok") for number, name, _ in ITEMS]
        report = ReadyReport(ts=datetime.now(UTC), config_hash=self.settings.config_hash(), items=items, **fields)
        dump(report, self.root / READY_PATH)

    def submitted(self, client, **kwargs):
        # a later process replays earlier submits with the secret hidden
        if kwargs["secrets"] != HIDDEN:
            self.secrets.append(kwargs["secrets"])
            self.push_record(kwargs["labels"]["driftmap_run"])
        return SUBMIT(client, **kwargs)

    def push(self, path, data):
        self.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=RESULTS, repo_type="dataset")

    def push_record(self, run_id, status="ok", **changes):
        changes.setdefault("config_hash", self.settings.config_hash())
        record = run_record(run_id, changes.pop("config_hash"), status, **changes)
        self.push(f"runs/{run_id}.json", record.model_dump_json().encode())

    def smoke_passed(self, kind):
        plan = build_plan(self.settings, kind, "full", None, SHA, repo_root=self.root)
        record = SmokeRecord(
            script=plan.script,
            image=plan.image,
            runner=plan.runner,
            dependencies=pins(self.root),
            schema_versions=schema_versions(),
            status="ok",
            bos_check=None,
            install_minutes=2.5,
            upload_mb_s=11.0,
            peak_vram_gb=9.4,
            peak_rss_gb=6.1,
            tokens_per_s=80.0,
        )
        dump(record, self.root / SMOKE_DIR / f"{kind}.json")

    def script(self, **facts):
        self.client_log.with_name(f"{self.client_log.name}.script.json").write_text(json.dumps(facts))

    def submits(self):
        if not self.client_log.exists():
            return []
        calls = [json.loads(line) for line in self.client_log.read_text().splitlines()]
        return [call["kwargs"]["command"][-1] for call in calls if call["method"] in ("run_job", "run_uv_job")]


@pytest.fixture(autouse=True)
def process_state(monkeypatch):
    """What the app sets for the whole process is put back."""
    root = logging.getLogger()
    names = ("", "httpx", "httpx2", "httpcore")
    handlers, factory = root.handlers[:], logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in names}
    monkeypatch.setattr(clients, "_backend", clients._backend)
    core_hub.token.cache_clear()
    yield
    root.handlers[:] = handlers
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    logging.setLogRecordFactory(factory)
    core_hub.token.cache_clear()


def jobs_pending(world):
    """Pending flag of each job on the ledger; holds are left out."""
    return [line.pending for job_id, line in world.ledger.read().items() if not job_id.startswith("unsubmitted:")]


@pytest.fixture
def world(monkeypatch):
    return World(monkeypatch, Path.cwd())


def verdict(result, guard):
    (row,) = [line.split() for line in result.output.splitlines() if line.startswith(f"{guard} ")]
    return row[1]


def test_help_lists_every_verb():
    assert sorted(typer.main.get_command(cli.app).commands) == VERBS
    result = run("--help")
    assert result.exit_code == 0 and all(f" {verb} " in result.output for verb in VERBS)
    names = ("data", "ledger", "table", "judge")
    groups = {name: sorted(typer.main.get_command(cli.app).commands[name].commands) for name in names}
    assert groups == {"data": ["build"], "ledger": ["reconcile"], "table": ["build"], "judge": ["closed", "reparse"]}
    assert sorted(typer.main.get_command(cli.app).commands["decide"].commands) == ["drift"]


def test_the_data_package_is_imported_only_inside_its_verb():
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    top = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    names = [alias.name for node in top for alias in node.names] + [getattr(node, "module", "") for node in top]
    assert not [name for name in names if name and name.startswith("driftmap.data")]


def test_a_dry_run_prints_the_plan_and_a_full_run_needs_its_smoke_record(world):
    """A dry run prints the plan with the token masked; without a smoke record a full run is refused (TS-PLN-2)."""
    smoke = run("launch", "generate", "--dry-run", *BASELINE)
    assert smoke.exit_code == 0
    assert smoke.output.splitlines()[0].split() == ["verb", "launch"]
    for expected in (
        f"image     {world.settings.images.pytorch}",
        "flavor    l4x1",
        "timeout   1800s (30 min)",
        "python -m driftmap.scripts.generate --run-id generate-smoke-",
        " --kind baseline\n",
        "secrets   HF_TOKEN=***",
        "estimate  399990 micro-USD ($0.40)",
        "ledger    0 micro-USD",
        "credit    20000000 micro-USD left",
    ):
        assert expected in smoke.output
    assert verdict(smoke, "smoke") == "pass"

    full = run("launch", "generate", "--mode", "full", "--dry-run", *BASELINE)
    assert full.exit_code == ExitCode.GUARD_REFUSAL == 4
    assert "timeout   5400s (90 min)" in full.output and "secrets   HF_TOKEN=***" in full.output
    assert verdict(full, "smoke") == "refused" and "no passing smoke run" in full.output
    assert "nothing was submitted" in full.output
    assert world.submits() == [] and not world.ledger.path.exists()


@examples(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tail=st.text(alphabet=string.ascii_letters + string.digits, min_size=20, max_size=40))
def test_the_token_reaches_the_job_as_a_secret_and_nothing_else(tail, tmp_path_factory, monkeypatch):
    """No output, log or local file of plan, dry run and two followed launches holds the token (TS-CFG-6)."""
    token = f"hf_{tail}"
    monkeypatch.setenv("HF_TOKEN", token)
    core_hub.token.cache_clear()
    world = World(monkeypatch, tmp_path_factory.mktemp("checkout"))

    results = [
        run("plan", "generate"),
        run("launch", "generate", "--dry-run", *BASELINE),
        run("-v", "launch", "generate", "--wait", *BASELINE),
        run("-v", "launch", "generate", "--mode", "full", "--wait", *BASELINE),
    ]

    assert [result.exit_code for result in results] == [0, 0, 0, 0]
    assert world.secrets == [{"HF_TOKEN": token}] * 2
    assert all("secrets   HF_TOKEN=***" in result.output for result in results)
    local = {
        path.relative_to(world.root / LOCAL).as_posix(): path.read_text()
        for path in (world.root / LOCAL).rglob("*")
        if path.is_file()
    }
    assert {"ready.json", "results/ledger.jsonl", "smoke/generate.json"} <= set(local)
    assert len([name for name in local if name.startswith("results/logs/")]) == 2
    assert jobs_pending(world) == [False, False]
    for text in (*(result.output for result in results), *local.values(), world.client_log.read_text()):
        assert token not in text


def test_a_drift_decision_is_stored_pushed_and_lifts_the_guard(world):
    """decide drift stores schema, decision, note and time; the drift guard then passes (TS-LCH-13)."""
    tables = Tables(
        schema=SCHEMA_TABLES,
        columns={},
        safety=[],
        over_refusal=[],
        chrf=[],
        deltas=[],
        flags={"drift_below_threshold": True},
        judge=judge_config(world.settings),
        seeds=Seeds(seed=1, bootstrap=2),
    )
    dump(tables, world.root / TABLES_PATH)
    before = run("launch", "repair", "--dry-run")
    assert before.exit_code == 4 and verdict(before, "drift_decision") == "refused"
    config_row = [
        line for line in run("plan", "drift", "--mode", "full").output.splitlines() if line.startswith("config ")
    ]

    done = run("decide", "drift", "--proceed", "--note", "x")

    assert done.exit_code == 0 and done.output.splitlines()[0].split() == ["verb", "decide", "drift"]
    stored = json.loads((world.root / DECISION_PATH).read_text())
    assert (stored["schema"], stored["decision"], stored["note"]) == ("driftmap.decision/1", "proceed", "x")
    assert abs(datetime.fromisoformat(stored["ts"]) - datetime.now(UTC)) < timedelta(minutes=1)
    assert world.hub.repos[RESULTS]["main"]["decisions/drift.json"] == (world.root / DECISION_PATH).read_bytes()
    after = run("launch", "repair", "--dry-run")
    assert after.exit_code == 0 and verdict(after, "drift_decision") == "pass"

    assert run("decide", "drift", "--extend", "--note", "one more epoch").exit_code == 0
    stored = json.loads(world.hub.repos[RESULTS]["main"]["decisions/drift.json"])
    assert (stored["decision"], stored["note"]) == ("extend", "one more epoch")
    again = [line for line in run("plan", "drift", "--mode", "full").output.splitlines() if line.startswith("config ")]
    assert again == config_row and len(config_row) == 2


def test_a_decision_needs_one_choice_a_note_and_a_clean_note(world):
    for argv in (
        ["--note", "x"],
        ["--proceed", "--extend", "--note", "x"],
        ["--proceed"],
        ["--proceed", "--note", " "],
    ):
        assert run("decide", "drift", *argv).exit_code == 2
    assert not (world.root / DECISION_PATH).exists()

    cut = f"hf_{'b' * 10}\n{'b' * 24}"
    for note in (f"see {HOME_DIR}/notes", f"token {cut}", "\x1b[31mred\x1b[0m", "account 123456789012"):
        leaked = run("decide", "drift", "--proceed", "--note", note)
        assert leaked.exit_code == ExitCode.REDACTION, note
        assert "b" * 10 not in leaked.output and HOME_DIR not in leaked.output
    assert "home_path" in run("decide", "drift", "--proceed", "--note", f"see {HOME_DIR}/notes").stderr
    assert "decisions/drift.json" not in world.hub.repos[RESULTS]["main"] and not (world.root / DECISION_PATH).exists()


def test_a_decision_that_cannot_be_pushed_exits_7_and_stays_local(world):
    world.hub.fail_after(1, 403)
    result = run("decide", "drift", "--proceed", "--note", "x")
    assert result.exit_code == ExitCode.PUSH_FAILED == 7
    assert "run the command again" in result.output and (world.root / DECISION_PATH).exists()


CODES = [
    ExitCode.USAGE,
    ExitCode.READINESS,
    ExitCode.GUARD_REFUSAL,
    ExitCode.LOST_RUN,
    ExitCode.REDACTION,
    ExitCode.DRIFT_BELOW_THRESHOLD,
]


@pytest.mark.parametrize("code", CODES)
def test_a_library_error_becomes_the_process_exit_code(world, monkeypatch, code):
    def stop(deps):
        raise DriftmapError(code, f"stopped with {TOKEN} in {HOME_DIR}/x")

    monkeypatch.setattr(launcher, "reconcile", stop)
    result = run("ledger", "reconcile")
    assert result.exit_code == int(code) and isinstance(result.exception, SystemExit)
    assert "error: stopped with hf_*** in <home>/x" in result.stderr and "Traceback" not in result.output


@pytest.mark.parametrize("error", [RuntimeError, OSError])
def test_an_unforeseen_error_prints_a_masked_traceback_and_exits_1(world, monkeypatch, error):
    def crash(deps):
        raise error(f"boom {TOKEN} in {HOME_DIR}/x")

    monkeypatch.setattr(launcher, "reconcile", crash)
    result = run("ledger", "reconcile")
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "Traceback" in result.stderr and f"{error.__name__}: boom hf_*** in <home>/x" in result.stderr
    assert TOKEN not in result.output and HOME_DIR not in result.output


def test_an_interrupted_launch_exits_130_after_the_recovery_text(world, monkeypatch):
    def interrupted(settings, deps, kind, mode, **flags):
        print("its hold stays on the ledger; run `driftmap wait unsubmitted:run-1`")
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "launch", interrupted)
    for argv in (["launch", "generate", *BASELINE], ["plan", "generate"]):
        result = run(*argv)
        assert result.exit_code == 130 and isinstance(result.exception, SystemExit)
        assert result.stdout.splitlines()[-1] == "its hold stays on the ledger; run `driftmap wait unsubmitted:run-1`"
        assert result.stderr.splitlines() == ["interrupted"] and "Traceback" not in result.output


def test_an_interrupt_while_following_a_job_names_the_wait_command(world, monkeypatch):
    def interrupted(client, job_id, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeJobsClient, "wait_for_job", interrupted)
    launched = run("launch", "generate", "--wait", *BASELINE)
    (job_id,) = [job_id for job_id in world.ledger.read() if not job_id.startswith("unsubmitted:")]
    again = run("wait", job_id)
    for result in (launched, again):
        assert result.exit_code == 130 and "Traceback" not in result.output
        assert result.stderr.splitlines() == [
            "interrupted",
            f"{job_id} is still pending; follow it with: driftmap wait {job_id}",
        ]
    assert world.ledger.read()[job_id].pending


def test_a_record_that_does_not_validate_exits_2_without_its_values(world):
    (world.root / DECISION_PATH).parent.mkdir(parents=True)
    (world.root / DECISION_PATH).write_text(json.dumps({"schema": "driftmap.decision/1", "decision": TOKEN}))
    for argv in (["plan", "repair"], ["launch", "repair"], ["chain", "repair"]):
        result = run(*argv)
        assert result.exit_code == 2 and "drift.json: decision: " in result.stderr, argv
        assert "hf_" not in result.output and "Traceback" not in result.output
    (world.root / DECISION_PATH).unlink()
    (world.root / TABLES_PATH).parent.mkdir(parents=True)
    (world.root / TABLES_PATH).write_text("{}")
    torn = run("plan", "repair")
    assert torn.exit_code == 2 and "tables.json: " in torn.stderr and world.submits() == []
    # a torn decision is replaced by deciding again
    (world.root / DECISION_PATH).write_text("{")
    assert run("decide", "drift", "--proceed", "--note", "x").exit_code == 0


def test_a_validation_error_past_the_file_loads_is_a_crash(world, monkeypatch):
    def broken(settings, deps, kind, mode, **flags):
        Decision.model_validate({"decision": "maybe"})

    monkeypatch.setattr(launcher, "launch", broken)
    result = run("plan", "drift")
    assert result.exit_code == 1 and "Traceback" in result.stderr and "ValidationError" in result.stderr


def test_usage_and_configuration_faults_exit_2(world):
    for argv in (
        ["nope"],
        ["launch"],
        ["launch", "nope"],
        ["launch", "generate", "--mode", "big"],
        ["plan", "generate", "--zzz"],
    ):
        assert run(*argv).exit_code == 2, argv
    for argv in (["chain"], ["wait"], ["scan"], ["scan", "absent.txt"], ["launch", "generate", "--limit", "x"]):
        assert run(*argv).exit_code == 2, argv
    assert run().exit_code == 2

    world.config.write_text("surprise = 1\n" + world.config.read_text())
    broken = run("plan", "generate")
    assert broken.exit_code == ExitCode.USAGE and "surprise" in broken.output
    world.config.unlink()
    assert run("ledger", "reconcile").exit_code == 2


def test_hub_verbs_stop_at_exit_3_before_a_client_exists(world, monkeypatch):
    monkeypatch.setattr(cli, "make_jobs_client", lambda settings: pytest.fail("a client was built"))
    (world.root / READY_PATH).unlink()
    verbs = (
        ["plan", "generate"],
        ["launch", "generate", *BASELINE],
        ["chain", "generate", "--target", "baseline"],
        ["decide", "drift", "--proceed", "--note", "x"],
        ["data", "build"],
        ["table", "build"],
        ["judge", "reparse", "--checkpoint", f"{AYA}@{'1' * 40}"],
    )
    for argv in verbs:
        result = run(*argv)
        assert result.exit_code == ExitCode.READINESS == 3, argv
        assert result.output.splitlines()[0].split()[0] == "verb" and "run `driftmap ready`" in result.stderr

    world.ready()
    world.config.write_text(world.config.read_text().replace("= 20261102", "= 7"))
    assert "another configuration" in run("plan", "generate").stderr


def test_launch_passes_its_flags_through_and_overrides_stay_off(world, monkeypatch):
    seen = []

    def launch(settings, deps, kind, mode, **flags):
        seen.append((kind, mode, flags, type(deps.ledger).__name__, deps.repo_root, deps.hub._api))
        return ExitCode.SUCCESS

    monkeypatch.setattr(launcher, "launch", launch)
    assert run("launch", "generate", *BASELINE).exit_code == 0
    assert run("plan", "steer", "--mode", "full").exit_code == 0
    argv = ["--mode", "full", "--wait", "--limit", "5", "--", "--checkpoint", "a/b@c"]
    assert run("launch", "quant_eval", *argv).exit_code == 0

    off = {"override_smoke": False, "override_soft": False}
    plain = {"wait": False, "dry_run": False, "limit": None, "args": ["--kind", "baseline"], **off}
    assert seen[0] == ("generate", "smoke", plain, "Ledger", world.root, world.hub)
    assert seen[1][:3] == ("steer", "full", {"dry_run": True})
    full = {"wait": True, "dry_run": False, "limit": 5, "args": ["--checkpoint", "a/b@c"], **off}
    assert seen[2][:3] == ("quant_eval", "full", full)


def test_generate_needs_its_job_argument_and_no_judge_job_can_be_launched(world, monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "launch", lambda settings, deps, kind, mode, **flags: seen.append(flags["args"]))
    with monkeypatch.context() as patch:
        patch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
        for argv in (["generate"], ["generate", "--dry-run"], ["generate", "--", "--checkpoint", f"{AYA}@{SHA}"]):
            missing = run("launch", *argv)
            assert missing.exit_code == 2, argv
            assert "driftmap launch generate --mode smoke -- --kind baseline|drifted|repaired" in flat(missing)
        checkpoint = ["--checkpoint", f"{AYA}@{SHA}"]
        for argv in (["judge"], ["judge", "--", *checkpoint], ["generate", "--after", "x", *BASELINE]):
            assert run("launch", *argv).exit_code == 2, argv
        assert run("plan", "judge").exit_code == 2
    assert seen == []

    assert run("launch", "generate", "--", "--kind=drifted").exit_code == 0
    assert run("launch", "drift").exit_code == 0
    assert seen == [["--kind=drifted"], []]


def test_job_arguments_cannot_restate_what_the_launcher_sets_or_carry_a_secret(world, monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
    owned = [
        "--run-id",
        "--timeout-s",
        "--config-hash",
        "--code-sha",
        "--submitted-at",
        "--limit",
        "--force-flush-after-s",
    ]
    for flag in (*owned, "--timeout", "--limit=5", "--sub"):
        result = run("launch", "drift", "--", "--epochs", "2", flag, "x")
        assert result.exit_code == 2 and f"{flag.split('=')[0]} is set by the launcher" in flat(result), flag
    cut = f"hf_{'b' * 10}\n{'b' * 24}"
    for secret in (TOKEN, cut, f"{HOME_DIR}/rows", "\x1b[31mred"):
        result = run("launch", "drift", "--", "--resume", secret)
        assert result.exit_code == 2 and "a job argument" in flat(result)
        assert "b" * 10 not in result.output and HOME_DIR not in result.output


def test_an_override_needs_a_reason_and_the_reason_lands_in_the_ledger(world, monkeypatch):
    with monkeypatch.context() as patch:
        # refused on the arguments alone
        patch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
        for flag in ("--override-smoke", "--override-soft"):
            for extra in ([], ["--reason", "  "], ["--reason", "\x1b[31m \x1b[0m\x07"]):
                refused = run("launch", "generate", "--mode", "full", flag, *extra, *BASELINE)
                assert refused.exit_code == 2 and "--reason" in refused.output
        assert run("launch", "generate", "--reason", "no override here", *BASELINE).exit_code == 2
    assert world.submits() == [] and not world.client_log.exists() and not world.ledger.path.exists()

    reason = f"same image as the smoke run,\n approved  in {HOME_DIR}/notes"
    done = run("launch", "generate", "--mode", "full", "--override-smoke", "--reason", reason, "--wait", *BASELINE)

    assert done.exit_code == 0 and len(world.submits()) == 1
    note = "reason=same image as the smoke run, approved in <home>/notes"
    hold, pending, released, last = world.ledger.lines()
    assert hold.pending and hold.job_id == f"unsubmitted:{pending.run_id}"
    assert hold.notes == ["price=hardware", "override-smoke", "hold", note]
    assert pending.pending and pending.notes == ["price=hardware", "override-smoke", note]
    assert (released.job_id, released.pending, released.notes[-1]) == (hold.job_id, False, f"job={pending.job_id}")
    assert not last.pending and last.job_id == pending.job_id and last.notes == pending.notes
    assert f"overrides --override-smoke {note.replace('=', ' ')}" in " ".join(done.output.split())

    cut = f"see hf_{'b' * 10}\n{'b' * 24}  now"
    refused = run("launch", "generate", "--mode", "full", "--override-soft", "--reason", cut, "--dry-run", *BASELINE)
    assert verdict(refused, "smoke") == "refused" and len(world.ledger.lines()) == 4
    assert "reason see hf_*** *** now config" in flat(refused) and "b" * 10 not in refused.output


def test_wait_follows_a_launched_job_and_an_unknown_job_exits_5(world):
    launched = run("launch", "generate", *BASELINE)
    assert launched.exit_code == 0, launched.output
    job_id = launched.output.split("driftmap wait ")[1].split()[0]
    assert world.ledger.read()[job_id].pending
    # a stale report must not keep a running job off the ledger
    (world.root / READY_PATH).unlink()

    waited = run("wait", job_id)
    assert waited.exit_code == 0 and waited.output.splitlines()[0].split() == ["verb", "wait"]
    assert not world.ledger.read()[job_id].pending and "next: none" in waited.output

    hold = f"unsubmitted:{world.ledger.read()[job_id].run_id}"
    closed = run("wait", hold)
    assert closed.exit_code == 0 and f"{hold} is already closed" in closed.output

    lost = run("wait", "job-unknown")
    assert lost.exit_code == ExitCode.LOST_RUN == 5 and "no ledger line for job job-unknown" in lost.stderr
    assert run("wait", "unsubmitted:nothing").exit_code == 5


def test_wait_takes_a_hold_id_and_neither_it_nor_reconcile_needs_readiness(world, monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "wait_job", lambda settings, deps, job_id: seen.append(job_id))
    world.config.write_text(world.config.read_text().replace("= 20261102", "= 7"))
    hold = "unsubmitted:generate-full-20261113T100000Z"
    for job_id in ("job-1", hold):
        result = run("wait", job_id)
        assert result.exit_code == 0 and result.output.splitlines()[:2] == ["verb      wait", f"job       {job_id}"]
    assert seen == ["job-1", hold]
    reconciled = run("ledger", "reconcile")
    assert reconciled.exit_code == 0 and reconciled.output.splitlines()[0].split() == ["verb", "ledger", "reconcile"]
    assert "billing 0 micro-USD, ledger 0 micro-USD" in reconciled.output


def test_reconcile_compares_billing_with_the_ledger(world):
    (world.root / READY_PATH).unlink()
    world.script(billing_used_micro_usd=400_000)
    result = run("ledger", "reconcile")
    assert result.exit_code == 0 and "billing 400000 micro-USD, ledger 0 micro-USD" in result.output

    world.script(billing_used_micro_usd=600_000)
    assert run("ledger", "reconcile").exit_code == 1


def test_outside_the_test_fake_a_verb_gets_the_real_clients(world, monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "wait_job", lambda settings, deps, job_id: seen.append(deps))
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    try:
        assert run("wait", "job-1").exit_code == 0
    finally:
        set_client_factory(default_client_factory)
    (deps,) = seen
    assert isinstance(deps.client, HfJobsClient) and isinstance(deps.client.api, HfApi)
    assert deps.hub._api is deps.client.api and deps.client.api.token == "hf_dummy"
    assert (deps.ledger.path, deps.repo_root) == (world.root / launcher.LEDGER_PATH, world.root)


def test_plan_uses_the_gpu_flavor_readiness_recorded(world):
    world.ready(gpu_flavor="a10g-small")
    result = run("plan", "generate")
    assert result.exit_code == 0 and "flavor    a10g-small" in result.output
    assert "estimate  500010 micro-USD" in result.output


def test_ready_stores_and_prints_the_report(world, monkeypatch):
    def fine(settings, deps):
        return Check(True, "fine")

    def closed(settings, deps):
        return Check(False, f"refused {TOKEN}", "ask for access")

    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", fine), (2, "second", fine)))
    passed = run("ready")
    assert passed.exit_code == 0 and passed.output.splitlines()[0].split() == ["verb", "ready"]
    assert " 1  PASS  first: fine" in passed.output and "secrets   HF_TOKEN=***" in passed.output
    assert run("plan", "generate").exit_code == 0

    as_json = run("ready", "--json")
    assert as_json.exit_code == 0 and "verb" in as_json.stderr
    assert json.loads(as_json.stdout) == json.loads((world.root / READY_PATH).read_text())

    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", fine), (2, "second", closed)))
    failed = run("ready")
    assert failed.exit_code == ExitCode.READINESS == 3
    assert " 2  FAIL  second: refused hf_***" in failed.output and "hint: ask for access" in failed.output
    assert "not ready: 2" in failed.output and run("plan", "generate").exit_code == 3
    failed_json = run("ready", "--json")
    assert failed_json.exit_code == 3 and json.loads(failed_json.stdout)["items"][1]["status"] == "FAIL"


def test_the_inputs_a_verb_prints_are_masked(world):
    result = run("wait", f"{HOME_DIR}/{TOKEN}")
    assert result.output.splitlines()[:2] == ["verb      wait", "job       <home>/hf_***"]
    assert TOKEN not in result.output and HOME_DIR not in result.output


def test_ready_pin_leaves_a_report_made_under_the_pinned_configuration(tmp_path, monkeypatch):
    root = tmp_path / "checkout-pin"
    root.mkdir()
    world = World(monkeypatch, root, revisions=None)
    for repo in (HARMFUL, BENIGN, XSTEST):
        world.hub.add_repo(repo, repo_type="dataset")
    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", lambda settings, deps: Check(True, "fine")),))
    unpinned = world.settings.config_hash()

    result = run("ready", "--pin")

    assert result.exit_code == 0
    pinned = json.loads((world.root / "revisions.json").read_text())
    assert pinned == {repo: world.hub.heads[repo] for repo in (AYA, HARMFUL, BENIGN, XSTEST)}
    assert "commit and push revisions.json" in result.output and all(repo in result.output for repo in pinned)
    stored = json.loads((world.root / READY_PATH).read_text())
    assert stored["config_hash"] == world.settings.config_hash() != unpinned
    assert run("plan", "generate").exit_code == 0

    again = run("ready", "--pin")
    assert again.exit_code == 0 and "already pins" in again.output
    assert json.loads((world.root / "revisions.json").read_text()) == pinned


def screen(settings, status):
    row = AlphaRow(
        alpha=0.25, safe_rate_pooled=0.5, n_judged=120, chrf_pooled=0.4, pass_safety=False, pass_chrf=True, error=None
    )
    return AlphaScreen(
        schema=SCHEMA_ALPHA_SCREEN,
        alphas=[row],
        chosen=None,
        recover_status=status,
        badge="Faithful",
        safetune_version="0.1.10",
        dare=Dare(),
        judge=judge_config(settings),
        bars=Bars(),
        baseline_safe_pooled=0.9,
        drifted_chrf_subset=0.4,
        tied_weight_owner="embed",
    )


def reader(world, mode="full", target=None):
    deps = Deps(client=None, hub=Hub(world.hub), ledger=world.ledger, repo_root=world.root)
    return cli._state(world.settings, deps, mode, target)


def test_the_chain_reader_takes_state_from_the_results_repo(world):
    settings = world.settings
    read = reader(world)
    bare = State(languages=tuple(settings.languages), checkpoint="drifted")
    assert read("generate") == bare
    assert reader(world, target="baseline")("generate").checkpoint == "baseline"

    world.push_record("drift-full-20261105T120000Z", config_hash="0" * 64, merged_sha256="a" * 64)
    world.push_record("drift-smoke-20261105T090000Z", merged_sha256="b" * 64)
    world.push_record("drift-full-20261105T100000Z", merged_sha256="c" * 64)
    world.push_record("drift-full-20261105T110000Z", merged_sha256="d" * 64)
    world.push_record("drift-full-20261106T100000Z", "failed")
    world.push_record("generate-full-20261106T100000Z")
    repaired = Ref(kind="model", repo=f"{OWNER}/tiny-aya-global-repaired", revision="7" * 40)
    world.push_record("repair-full-20261107T100000Z", outputs=[repaired])
    gguf = Ref(kind="model", repo=f"{OWNER}/tiny-aya-global-repaired-GGUF", revision="8" * 40)
    world.push_record("quant-full-20261108T100000Z", outputs=[gguf])
    world.push("runs/torn.json", b"{")
    world.push("alpha_screen.json", screen(settings, "failed").model_dump_json().encode())

    full = State(
        languages=tuple(settings.languages),
        checkpoint="drifted",
        merged_sha256="d" * 64,
        repaired_revision="7" * 40,
        gguf_record=True,
        recover_status="failed",
    )
    assert read("quant") == full
    # the screen counts only beside an ok repair run of this mode and config
    smoke = reader(world, "smoke")("steer")
    assert smoke == State(languages=tuple(settings.languages), checkpoint="drifted", merged_sha256="b" * 64)
    world.push_record("repair-full-20261107T100000Z", config_hash="0" * 64, outputs=[repaired])
    assert read("steer").recover_status is None and read("quant").repaired_revision is None
    world.push_record("repair-full-20261107T110000Z", outputs=[repaired])
    assert read("steer").recover_status == "failed"
    world.push("alpha_screen.json", b"{")
    assert read("steer").recover_status is None



def test_chain_names_its_generate_target_and_takes_no_judge_step(world, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
        for argv, hint in (
            (["generate", "steer"], "--target baseline|drifted|repaired"),
            (["drift", "--target", "baseline"], "--target names the generate step"),
            (["generate", "generate", "--target", "baseline"], "one generate step"),
            (["judge"], "judge"),
            (["generate", "judge", "--target", "baseline"], "judge"),
            (["generate", "--target", "merged"], "merged"),
        ):
            refused = run("chain", *argv)
            assert refused.exit_code == 2 and hint in flat(refused), argv

    blocked = run("chain", "generate", "steer", "--target", "drifted")
    assert blocked.exit_code == 4 and blocked.output.splitlines()[0].split() == ["verb", "chain"]
    assert "generate cannot run yet; missing: drift record with merged_sha256" in blocked.output
    repair = run("chain", "repair")
    assert repair.exit_code == 4 and "baseline labels for en, fr, es, ru, hi, ar" in repair.output
    assert world.submits() == []

    world.smoke_passed("generate")
    done = run("chain", "generate", "--target", "baseline")
    assert done.exit_code == 0, done.output
    (only,) = world.submits()
    assert "driftmap.scripts.generate --run-id generate-full-" in only and only.endswith(" --kind baseline")
    assert jobs_pending(world) == [False]


def test_chain_generates_the_drifted_target_once_its_drift_record_exists(world):
    world.smoke_passed("generate")
    world.push_record("drift-full-20261105T100000Z", merged_sha256="c" * 64)
    drifted = run("chain", "generate", "--target", "drifted")
    assert drifted.exit_code == 0, drifted.output
    assert world.submits()[-1].endswith(" --kind drifted")


def test_scan_exits_6_on_a_finding_and_walks_directories(world):
    clean = world.root / "notes.txt"
    clean.write_text("nothing here\n")
    passed = run("scan", str(clean))
    assert passed.exit_code == 0 and passed.output.splitlines()[0].split() == ["verb", "scan"]
    assert "no finding in 1 file" in passed.output

    folder = world.root / "out"
    (folder / "deep").mkdir(parents=True)
    (folder / "deep" / "log.txt").write_text(f"ok\ntoken {TOKEN}\n")
    rows = world.root / LOCAL / "data" / "prompts" / "harmful"
    rows.mkdir(parents=True)
    (rows / "en.jsonl").write_text('{"prompt": "call 123456789012"}\n')
    (world.root / "ids.txt").write_text("123456789012\n")

    found = run("scan", "notes.txt", "out", str(world.root / LOCAL / "data"), "ids.txt")
    assert found.exit_code == ExitCode.REDACTION == 6
    assert "out/deep/log.txt:2: hf_token" in found.stderr and "ids.txt:1: account_id" in found.stderr
    assert "en.jsonl" not in found.stderr and TOKEN not in found.output


def test_a_verb_logs_through_the_masking_handler_and_v_turns_on_debug(world, monkeypatch):
    def noisy(deps):
        logging.getLogger("driftmap.test").warning("token %s", TOKEN)
        logging.getLogger("driftmap.test").debug("fine detail")
        return ExitCode.SUCCESS

    monkeypatch.setattr(launcher, "reconcile", noisy)
    quiet = run("ledger", "reconcile")
    assert "token hf_***" in quiet.stderr and TOKEN not in quiet.output and "fine detail" not in quiet.output
    assert clients._backend.__name__ == "HfJobsClient"
    loud = run("-v", "ledger", "reconcile")
    assert "fine detail" in loud.stderr and logging.getLogger().level == logging.DEBUG


def test_the_evaluation_pin_follows_the_upload_and_moves_only_on_request(world, monkeypatch):
    path = world.root / "revisions.json"
    sources = json.loads(path.read_text())

    def built(commit, uploaded):
        made = data_build.Built(SimpleNamespace(per_language={}, note="none"), commit, uploaded)
        monkeypatch.setattr(data_build, "build", lambda *args, **flags: made)

    def pin():
        return json.loads(path.read_text()).get(EVAL)

    assert run("data", "build", "--dry-run", "--repin").exit_code == 2
    # rows already on the Hub and no pin yet: the head the upload answered with
    built("a" * 40, False)
    first = run("data", "build")
    assert first.exit_code == 0 and f"pinned {EVAL} at {'a' * 40}\ncommit and push" in first.output
    assert json.loads(path.read_text()) == {**sources, EVAL: "a" * 40} and "stop counting" not in first.output
    world.ready()

    # labels landed since: the head moved, nothing was uploaded, the pin stays
    built("b" * 40, False)
    kept = run("data", "build")
    assert kept.exit_code == 0 and f"{EVAL} stays pinned at {'a' * 40}" in kept.output and pin() == "a" * 40
    assert "nothing was uploaded" in kept.output

    built("c" * 40, True)
    refused = run("data", "build")
    assert refused.exit_code == ExitCode.GUARD_REFUSAL == 4 and pin() == "a" * 40
    assert "driftmap data build --repin" in refused.stderr and "stop counting" in refused.stderr
    assert f"uploaded to {EVAL}" in refused.output

    # the rerun uploads nothing; the request alone moves the pin
    built("c" * 40, False)
    moved = run("data", "build", "--repin")
    assert moved.exit_code == 0 and pin() == "c" * 40 and json.loads(path.read_text()) == {**sources, EVAL: "c" * 40}
    assert f"pinned {EVAL} at {'c' * 40}" in moved.output and f"it was {'a' * 40}" in moved.output
    assert "stop counting" in moved.output and "run `driftmap ready` again" in moved.output
    world.ready()
    same = run("data", "build", "--repin")
    assert same.exit_code == 0 and f"{EVAL} is already pinned at {'c' * 40}" in same.output


def test_a_data_dry_run_prints_the_counts_and_touches_nothing(tmp_path, monkeypatch):
    fake = source_hub(
        tmp_path / "hub",
        harmful={iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX},
        train=[row for iso3 in SIX for row in benign_rows(iso3, 7)],
        test=benign_rows("eng", 3, "test") + benign_rows("arb", 3, "test"),
        xstest=fixture_rows("xstest_sample.jsonl"),
    )
    root = tmp_path / "checkout-data"
    root.mkdir()
    world = World(monkeypatch, root, hub=fake, n_harmful=12, n_heldout=12, revisions={BASE: None, **heads(fake)})
    pinned = (root / "revisions.json").read_bytes()
    # the first contact with the sources comes before any readiness report
    (root / READY_PATH).unlink()

    dry = run("data", "build", "--dry-run")

    assert dry.exit_code == 0, dry.output
    assert dry.output.splitlines()[0].split() == ["verb", "data", "build"]
    manifest = json.loads((root / LOCAL / "data" / "manifest.json").read_text())
    for language, counts in manifest["per_language"].items():
        row = f"{language}  harmful {counts['harmful']}  train {counts['benign_train']}"
        assert f"{row}  held-out {counts['benign_heldout']}" in dry.output
    assert sorted(manifest["per_language"]) == sorted(world.settings.languages)
    assert f"note: {manifest['note']}" in dry.output and "nothing was uploaded or pinned" in dry.output
    assert not {"upload_folder", "upload_file", "create_repo"} & {name for name, _ in fake.calls}
    assert (root / "revisions.json").read_bytes() == pinned

    assert run("data", "build").exit_code == ExitCode.READINESS
    assert not {"upload_folder", "upload_file", "create_repo"} & {name for name, _ in fake.calls}
    world.ready()
    built = run("data", "build")
    assert built.exit_code == 0, built.output
    assert f"uploaded to {EVAL}" in built.output and "manifest.json" in fake.repos[EVAL]["main"]
    # jobs read the rows at the revision the upload made
    assert json.loads((root / "revisions.json").read_text()) == {**heads(fake), EVAL: fake.heads[EVAL]}
    assert world.settings.revisions[EVAL] == fake.heads[EVAL]
    assert f"pinned {EVAL} at {fake.heads[EVAL]}" in built.output
    assert "commit and push revisions.json, then run `driftmap ready` again" in built.output
    # the new pin is part of the config the report was made under
    assert run("plan", "generate").exit_code == ExitCode.READINESS

    world.ready()
    uploaded = fake.heads[EVAL]
    again = run("data", "build")
    assert again.exit_code == 0 and f"{EVAL} is already pinned at {uploaded}" in again.output
    assert "nothing was uploaded" in again.output

    # a judge run commits labels beside the rows; a rebuild leaves the pin and the config hash alone
    fake.upload_file(path_or_fileobj=b"{}\n", path_in_repo="labels/x/en.jsonl", repo_id=EVAL, repo_type="dataset")
    later = run("data", "build")
    assert later.exit_code == 0 and f"{EVAL} stays pinned at {uploaded}" in later.output
    assert json.loads((root / "revisions.json").read_text())[EVAL] == uploaded != fake.heads[EVAL]
    assert run("plan", "drift").exit_code == 0


BENIGN_PIN = "4" * 40
TABLES_DIR = TABLES_PATH.parent


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


SMOKE_DRIFTED = fake_labels.CHECKPOINTS["drifted"].replace("@", "-smoke@")


def put_labels(world, rows, mode="full"):
    files = {}
    for row in rows:
        path = ids.labels_path(ids.slug(row.checkpoint), row.language, mode, benign=row.set == "benign")
        files[path] = files.get(path, b"") + row.model_dump_json(by_alias=True).encode() + b"\n"
    for path, data in files.items():
        world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")


def generated(world, column, checkpoint=None, **changes):
    """Record the generate run that read ``checkpoint`` (default: the column's) on the results dataset."""
    repo, _, revision = (checkpoint or fake_labels.CHECKPOINTS[column]).partition("@")
    fields = {"run_id": f"generate-{changes.get('mode', 'full')}-{column}", "config_hash": world.settings.config_hash()}
    record = fake_labels.run_record(changes.pop("kind", "generate"), column, **{**fields, **changes})
    record.inputs[0] = Ref(kind="model", repo=repo, revision=revision)
    world.push(f"runs/{record.run_id}.json", record.model_dump_json(by_alias=True).encode())
    return record


def evaluation(world, columns, mode="full", n=4, listed=None):
    """Put prompt files, training rows, and per column labels and their generate run on the fake Hub."""
    if EVAL not in world.hub.repos:
        world.hub.add_repo(EVAL, repo_type="dataset")
    for column, unsafe in columns.items():
        # a smoke drift run pushes to its own repos
        changes = {"checkpoint": SMOKE_DRIFTED} if (column, mode) == ("drifted", "smoke") else {}
        put_labels(world, fake_labels.column_rows(column, n=n, unsafe=unsafe, **changes), mode)
        generated(world, column, changes.get("checkpoint"), mode=mode)
    for name, by_language in fake_labels.targets(listed or n).items():
        folder = {"benign": "benign_heldout", "train": "sft_train"}.get(name, name)
        for language, prompt_ids in by_language.items():
            rows = [{"prompt_id": prompt_id, "prompt": f"placeholder {prompt_id}"} for prompt_id in prompt_ids]
            if name == "train":
                rows = [{"prompt": [{"role": "user", "content": row["prompt"]}], "completion": []} for row in rows]
            path = f"{folder}/{language}.jsonl" if name == "train" else f"prompts/{folder}/{language}.jsonl"
            world.hub.upload_file(path_or_fileobj=jsonl(rows), path_in_repo=path, repo_id=EVAL, repo_type="dataset")


def stored(world, path):
    return world.hub.repos[RESULTS]["main"][path]


def test_table_build_reads_the_hub_writes_prints_and_uploads_the_tables(world):
    absent = run("table", "build")
    assert absent.exit_code == 1 and f"{EVAL} could not be read" in absent.stderr and "Traceback" not in absent.output
    evaluation(world, {})
    unjudged = run("table", "build")
    assert unjudged.exit_code == 2 and "no judged labels" in unjudged.stderr and not (world.root / TABLES_PATH).exists()
    evaluation(world, {"baseline": 0, "drifted": 2})

    built = run("table", "build")

    assert built.exit_code == 0, built.output
    assert built.output.splitlines()[:2] == ["verb      table build", "mode      full"]
    local = world.root / TABLES_PATH
    tables = Tables.model_validate_json(local.read_bytes())
    states = {column: state.state for column, state in tables.columns.items()}
    assert states == {"baseline": "tabled", "drifted": "tabled", "repaired": "not_run", "repaired_q4": "not_run"}
    assert tables.flags["drift_below_threshold"] is False and tables.safety[0].n == 4
    shown = render.render(tables, world.settings)
    assert built.output.index("# Safety drift map") < built.output.index("written to .run-local/results/tables")
    assert (world.root / TABLES_DIR / "tables.md").read_text(encoding="utf-8") == shown and shown in built.output
    assert stored(world, "tables/tables.json") == local.read_bytes()
    assert stored(world, "tables/tables.md") == shown.encode()
    assert not any(path.startswith("smoke/") for path in world.hub.repos[RESULTS]["main"])

    # run records are read: a generation run that says so makes its column not comparable
    generated(world, "drifted", notes=[f"{NOT_COMPARABLE}another \x1b[31mseed"])
    world.push("runs/torn.json", b"{")
    world.push("alpha_screen.json", screen(world.settings, "failed").model_dump_json().encode())
    again = run("table", "build")
    # printed masked: the colour code a record carried is gone
    assert again.exit_code == 0 and "drifted ▨ (weights): not comparable: another  seed" in again.output
    assert "\x1b" not in again.output
    # the alpha screen is read too: a failed repair leaves nothing to quantize
    assert "- repaired_q4 (q4_k_m): not applicable" in again.output

    pins = json.loads((world.root / "revisions.json").read_text())
    (world.root / "revisions.json").write_text(json.dumps({repo: pin for repo, pin in pins.items() if repo != BENIGN}))
    world.ready()
    unpinned = run("table", "build")
    assert unpinned.exit_code == 2 and f"{BENIGN} is not pinned" in unpinned.stderr


def test_table_build_takes_each_column_from_its_latest_ok_generate_run(world):
    evaluation(world, {"baseline": 0, "drifted": 2})
    later = fake_labels.CHECKPOINTS["drifted"].replace("a" * 40, "e" * 40)
    put_labels(world, fake_labels.column_rows("drifted", unsafe=1, checkpoint=later))

    def drifted():
        tables = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
        return {row.checkpoint for row in tables.safety if "drift" in row.checkpoint}

    # labels of a revision no finished run generated are left on the Hub
    assert run("table", "build").exit_code == 0 and drifted() == {fake_labels.CHECKPOINTS["drifted"]}
    newer = datetime(2026, 11, 14, tzinfo=UTC)
    for changes in ({"status": "failed"}, {"mode": "smoke"}, {"kind": "judge"}):
        generated(world, "drifted", later, run_id="generate-full-again", started=newer, **changes)
        assert run("table", "build").exit_code == 0 and drifted() == {fake_labels.CHECKPOINTS["drifted"]}, changes
    generated(world, "drifted", later, run_id="generate-full-again", started=newer)
    assert run("table", "build").exit_code == 0 and drifted() == {later}


def test_a_rebuild_that_can_no_longer_evaluate_the_flag_keeps_it_and_the_guard(world):
    evaluation(world, {"baseline": 0, "drifted": 0})
    assert run("table", "build").exit_code == 8
    local, on_hub = (world.root / TABLES_PATH).read_bytes(), stored(world, "tables/tables.json")
    generated(world, "drifted", notes=[f"{NOT_COMPARABLE}another seed"])

    again = run("table", "build")

    assert again.exit_code == ExitCode.DRIFT_BELOW_THRESHOLD and "driftmap decide drift" in again.stderr
    assert (world.root / TABLES_PATH).read_bytes() == local and stored(world, "tables/tables.json") == on_hub
    assert "# Safety drift map" not in again.output
    assert run("launch", "repair", "--dry-run").exit_code == 4
    assert run("decide", "drift", "--extend", "--note", "rerun the drift").exit_code == 0
    decided = run("table", "build")
    assert (
        decided.exit_code == 0
        and "drift_below_threshold" not in json.loads(stored(world, "tables/tables.json"))["flags"]
    )


def test_table_build_names_the_file_it_cannot_read(world):
    evaluation(world, {"baseline": 0, "drifted": 2})
    local = world.root / TABLES_PATH
    local.parent.mkdir(parents=True)
    # a torn earlier table holds no flag and is replaced
    local.write_text("{")
    assert run("table", "build").exit_code == 0 and Tables.model_validate_json(local.read_bytes())

    (world.root / DECISION_PATH).parent.mkdir(parents=True)
    (world.root / DECISION_PATH).write_text("{")
    torn = run("table", "build")
    assert torn.exit_code == 2 and "drift.json: " in torn.stderr
    (world.root / DECISION_PATH).unlink()

    for bad, where in ((b"{\n", "harmful/en.jsonl:1"), (b'{"prompt_id": "x"}\n[]\n', "harmful/en.jsonl:2")):
        world.hub.upload_file(
            path_or_fileobj=bad, path_in_repo="prompts/harmful/en.jsonl", repo_id=EVAL, repo_type="dataset"
        )
        shaped = run("table", "build")
        assert shaped.exit_code == ExitCode.INTEGRITY and where in shaped.stderr, where
    listed = jsonl([{"prompt_id": prompt_id} for prompt_id in fake_labels.prompt_ids("harmful", "en", 4)])
    for path, data in (("prompts/harmful/en.jsonl", listed), ("sft_train/en.jsonl", b"{}\n")):
        world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")
    assert "sft_train/en.jsonl:1" in run("table", "build").stderr

    del world.hub.repos[RESULTS]
    gone = run("table", "build")
    assert gone.exit_code == 1 and f"{RESULTS} could not be read (RepositoryNotFoundError)" in gone.stderr
    for result in (torn, shaped, gone):
        assert "Traceback" not in result.output


def test_a_new_drift_flag_exits_8_after_the_table_is_written_shown_and_uploaded(world):
    evaluation(world, {"baseline": 0, "drifted": 0})

    flagged = run("table", "build")

    assert flagged.exit_code == ExitCode.DRIFT_BELOW_THRESHOLD == 8 and "driftmap decide drift" in flagged.stderr
    assert "- drift_below_threshold: yes" in flagged.output
    assert json.loads(stored(world, "tables/tables.json"))["flags"]["drift_below_threshold"] is True
    assert "tables/tables.md" in world.hub.repos[RESULTS]["main"]
    refused = run("launch", "repair", "--dry-run")
    assert refused.exit_code == 4 and verdict(refused, "drift_decision") == "refused"
    # the flag is on file now; deciding, not rebuilding, lifts the guard
    assert run("table", "build").exit_code == 0
    assert run("decide", "drift", "--proceed", "--note", "small n").exit_code == 0
    assert run("table", "build").exit_code == 0 and run("launch", "repair", "--dry-run").exit_code == 0
    # a flag new to this checkout, already decided on
    (world.root / TABLES_PATH).unlink()
    assert run("table", "build").exit_code == 0


def test_a_smoke_table_reads_and_writes_its_own_tree_and_leaves_the_guard_alone(world):
    # six prompts listed, the smoke runs took the first four
    evaluation(world, {"baseline": 0, "drifted": 0}, mode="smoke", listed=6)
    assert "no judged labels" in run("table", "build").stderr

    built = run("table", "build", "--mode", "smoke")

    assert built.exit_code == 0, built.output
    local = world.root / LOCAL / "results" / "smoke" / "tables"
    tables = Tables.model_validate_json((local / "tables.json").read_bytes())
    assert tables.flags["drift_below_threshold"] is True and tables.safety[0].n == 4
    assert stored(world, "smoke/tables/tables.json") == (local / "tables.json").read_bytes()
    assert stored(world, "smoke/tables/tables.md") == (local / "tables.md").read_bytes()
    assert not (world.root / TABLES_PATH).exists() and "tables/tables.json" not in world.hub.repos[RESULTS]["main"]
    assert run("launch", "repair", "--dry-run").exit_code == 0

    # a column that stopped short of the others is not complete
    short = fake_labels.column_rows("drifted", n=3, checkpoint=SMOKE_DRIFTED)
    path = ids.labels_path(ids.slug(short[0].checkpoint), "en", "smoke")
    rows = [row for row in short if row.language == "en" and row.set != "benign"]
    data = "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()
    world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")
    cut = run("table", "build", "--mode", "smoke")
    assert cut.exit_code == ExitCode.INTEGRITY and "drifted harmful/en: 3 rows for 4 prompts" in cut.stderr


def test_table_build_refuses_a_training_prompt_among_the_evaluation_prompts(world):
    leaked = ids.prompt_id(world.settings.sources.benign, BENIGN_PIN, "placeholder train-en-000")
    assert world.settings.revisions[world.settings.sources.benign] == BENIGN_PIN
    evaluation(world, {"baseline": 0})
    listed = [{"prompt_id": prompt_id} for prompt_id in (leaked, *fake_labels.prompt_ids("xstest", "en", 4)[1:])]
    world.hub.upload_file(
        path_or_fileobj=jsonl(listed), path_in_repo="prompts/xstest/en.jsonl", repo_id=EVAL, repo_type="dataset"
    )
    result = run("table", "build")
    assert result.exit_code == ExitCode.INTEGRITY == 12
    assert f"train prompt {leaked} is also a xstest prompt" in result.stderr
    assert not (world.root / TABLES_PATH).exists()


def test_tables_that_cannot_be_uploaded_stay_local_and_exit_7(world, monkeypatch):
    evaluation(world, {"baseline": 0, "drifted": 0})
    monkeypatch.setattr(world.hub, "upload_folder", lambda **call: world.hub._raise(403, "https://huggingface.co/x"))
    result = run("table", "build")
    assert result.exit_code == ExitCode.PUSH_FAILED == 7 and "run the command again" in result.stderr
    assert (world.root / TABLES_PATH).exists() and "# Safety drift map" in result.output
    # the flag this build raised is said before the upload could fail
    assert "driftmap decide drift" in result.stderr


def test_a_finding_in_the_tables_stops_the_print_and_the_upload(world):
    evaluation(world, {"baseline": 0, "drifted": 2})
    generated(world, "drifted", notes=[f"{NOT_COMPARABLE}weights read from {HOME_DIR}/merged"])
    (world.root / TABLES_DIR).mkdir(parents=True)
    (world.root / TABLES_DIR / "notes.txt").write_text("kept local")
    found = run("table", "build")
    assert found.exit_code == ExitCode.REDACTION and "home_path" in found.stderr
    assert "# Safety drift map" not in found.output and HOME_DIR not in found.output
    assert not [path for path in world.hub.repos[RESULTS]["main"] if path.startswith("tables/")]

    generated(world, "drifted")
    assert run("table", "build").exit_code == 0
    assert sorted(path for path in world.hub.repos[RESULTS]["main"] if path.startswith("tables/")) == [
        "tables/tables.json",
        "tables/tables.md",
    ]


def test_a_table_that_cannot_be_saved_is_not_shown_or_uploaded(world, monkeypatch):
    evaluation(world, {"baseline": 0, "drifted": 2})

    def refuse(tables, root, decisions):
        raise DriftmapError(ExitCode.INTEGRITY, "the tables cannot be stored")

    monkeypatch.setattr(table_build, "save", refuse)
    result = run("table", "build")
    assert result.exit_code == ExitCode.INTEGRITY and "# Safety drift map" not in result.output
    assert "tables/tables.json" not in world.hub.repos[RESULTS]["main"]


def test_judge_reparse_hands_the_checkpoint_to_the_library(world, monkeypatch):
    seen = []

    def reparsed(hub, labels_repo, raw_repo, checkpoint, mode):
        seen.append((hub._api, labels_repo, raw_repo, checkpoint, mode))
        return [] if mode == "smoke" else ["labels/x/superseded/h/en.jsonl", "labels/x/en.jsonl"]

    monkeypatch.setattr(reparse, "reparse_checkpoint", reparsed)
    checkpoint = f"{AYA}@{'1' * 40}"
    for mode in ("full", "smoke"):
        evaluation(world, {"baseline": 0}, mode=mode)
    done = run("judge", "reparse", "--checkpoint", checkpoint)
    assert done.exit_code == 0 and done.output.splitlines()[0].split() == ["verb", "judge", "reparse"]
    assert "labels/x/en.jsonl" in done.output and "2 files uploaded" in done.output
    same = run("judge", "reparse", "--checkpoint", checkpoint, "--mode", "smoke")
    assert same.exit_code == 0 and "nothing to rewrite" in same.output
    responses = f"{OWNER}/safety-drift-map-responses"
    assert seen == [(world.hub, EVAL, responses, checkpoint, mode) for mode in ("full", "smoke")]

    for bad in (AYA, "not a reference", f"{AYA}@{'1' * 40}#file"):
        assert run("judge", "reparse", "--checkpoint", bad).exit_code == 2
    assert run("judge", "reparse").exit_code == 2 and len(seen) == 2

    other = f"{AYA}@{'9' * 40}"
    unread = run("judge", "reparse", "--checkpoint", other, "--mode", "smoke")
    assert unread.exit_code == 2 and f"no labels for {other} in the smoke tree" in unread.stderr and len(seen) == 2


ROUTE = "gateway-x/model-9"


def test_judge_closed_hands_its_options_to_the_library_and_never_shows_the_route(world, monkeypatch):
    seen = []

    class Client:
        prompt_tokens, completion_tokens = 1400, 140

    def judged(settings, hub, client, checkpoint, **options):
        options.pop("echo")("harmful/en   scored 10  unscored 0  safe 7  refused 6  new 10")
        seen.append((hub._api, type(client), checkpoint, options))

    monkeypatch.setattr(closed, "judge_checkpoint", judged)
    monkeypatch.setattr(closed, "LiteLLMCompleter", lambda route: seen.append(route) or Client())
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    checkpoint = f"{OWNER}/tiny-aya-global-repaired-GGUF@{'2' * 40}#model-q4_k_m.gguf"
    (world.root / READY_PATH).unlink()

    done = run("judge", "closed", "--checkpoint", checkpoint)

    assert done.exit_code == 0 and done.output.splitlines()[0].split() == ["verb", "judge", "closed"]
    defaults = {"sets": ["harmful", "xstest"], "mode": "smoke", "concurrency": 4, "limit": None}
    defaults["retry_unscored"] = False
    assert seen == [ROUTE, (world.hub, Client, checkpoint, defaults)]
    assert "judge     closed-model-a under rubric-v1 (" in done.output
    assert "scored 10" in done.output and "tokens    1400 in, 140 out" in done.output
    assert ROUTE not in done.output and "gateway" not in done.output

    options = ["--mode", "full", "--sets", "xstest", "--concurrency", "2", "--limit", "5", "--retry-unscored"]
    assert run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", *options).exit_code == 0
    passed = {"sets": ["xstest"], "mode": "full", "concurrency": 2, "limit": 5, "retry_unscored": True}
    assert seen[-1][2:] == (f"{AYA}@{'1' * 40}", passed)


def test_judge_closed_takes_a_rows_path_in_place_of_a_checkpoint(world, monkeypatch):
    seen = []

    class Client:
        prompt_tokens, completion_tokens = 0, 0

    monkeypatch.setattr(closed, "judge_checkpoint", lambda *args, **options: pytest.fail("the checkpoint form ran"))
    monkeypatch.setattr(closed, "judge_rows", lambda settings, hub, client, pattern, **options: seen.append(
        (hub._api, pattern, {name: value for name, value in options.items() if name != "echo"})
    ))
    monkeypatch.setattr(closed, "LiteLLMCompleter", lambda route: Client())
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    path = "screen/tiny-aya-global-drift-merged--444444444444/alpha-*.jsonl"

    done = run("judge", "closed", "--rows", path, "--mode", "full", "--limit", "40", "--retry-unscored")

    assert done.exit_code == 0 and f"rows      {path}" in done.output and "checkpoint" not in done.output
    assert seen == [(world.hub, path, {"mode": "full", "concurrency": 4, "limit": 40, "retry_unscored": True})]
    for bad in (
        ["--rows", path, "--checkpoint", f"{AYA}@{'1' * 40}"],
        ["--rows", path, "--sets", "harmful"],
        ["--rows", f"smoke/{path}"],
        ["--rows", "screen/../labels/x.jsonl"],
        ["--rows", "/screen/x.jsonl"],
        ["--rows", "screen/x/alpha-0.1.json"],
    ):
        assert run("judge", "closed", *bad).exit_code == 2, bad
    assert len(seen) == 1


def test_judge_closed_refuses_bad_options_a_missing_route_and_a_missing_section(world, monkeypatch):
    monkeypatch.setattr(closed, "judge_checkpoint", lambda *args, **options: pytest.fail("the library was called"))
    checkpoint = ["--checkpoint", f"{AYA}@{'1' * 40}"]
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    for bad in (
        ["--checkpoint", AYA],
        ["--checkpoint", "not a reference"],
        [*checkpoint, "--sets", "benign"],
        [*checkpoint, "--sets", ","],
        [*checkpoint, "--concurrency", "0"],
        [*checkpoint, "--concurrency", "64"],
        [*checkpoint, "--limit", "0"],
        [*checkpoint, "--mode", "fast"],
        [],
    ):
        assert run("judge", "closed", *bad).exit_code == 2, bad

    monkeypatch.delenv(closed.ROUTE_ENV)
    unset = run("judge", "closed", *checkpoint)
    assert unset.exit_code == 2 and closed.ROUTE_ENV in flat(unset)

    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    World(monkeypatch, world.root, hub=world.hub, judge={"closed": None})
    absent = run("judge", "closed", *checkpoint)
    assert absent.exit_code == 2 and "[judge.closed]" in flat(absent) and ROUTE not in absent.output


def test_a_throttled_closed_judge_run_exits_4_with_its_message_and_the_token_count(world, monkeypatch):
    class Client:
        prompt_tokens, completion_tokens = 700, 70

    def throttled(*args, **options):
        raise closed.Throttled("the judge route is throttling or out of quota; run the command again")

    monkeypatch.setattr(closed, "judge_checkpoint", throttled)
    monkeypatch.setattr(closed, "LiteLLMCompleter", lambda route: Client())
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)

    stopped = run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}")

    assert stopped.exit_code == 4 and "throttling or out of quota; run the" in flat(stopped)
    assert "tokens    700 in, 70 out" in stopped.output and ROUTE not in stopped.output


def test_status_shows_spend_pending_jobs_and_the_latest_runs_without_writing(world):
    world.script(billing_used_micro_usd=400_000)
    assert run("launch", "generate", *BASELINE).exit_code == 0
    (job_id,) = [job_id for job_id in world.ledger.read() if not job_id.startswith("unsubmitted:")]
    world.push_record("drift-full-20261112T100000Z", minutes=171, cost_micro_usd=2_279_943)
    world.push_record("drift-full-20261111T100000Z", "failed", started=datetime(2026, 11, 1, tzinfo=UTC))
    world.push("runs/torn.json", b"{")
    (world.root / READY_PATH).unlink()
    ledger, calls = world.ledger.path.read_bytes(), len(world.hub.calls)

    shown = run("status")

    assert shown.exit_code == 0 and shown.output.splitlines()[0].split() == ["verb", "status"]
    text = flat(shown)
    assert "ledger 399990 micro-USD ($0.40)" in text
    assert f"pending {job_id} generate (smoke) held at 399990 micro-USD" in text
    assert "billing 400000 micro-USD ($0.40) used, 19600000 left of the credit" in text
    assert "headroom 15600000 micro-USD ($15.60) under the soft ceiling" in text
    assert [line.split()[1] for line in shown.output.splitlines() if line.startswith("pending")] == [job_id]
    assert "drift full ok 171 2279943 drift-full-20261112T100000Z" in text
    assert "generate smoke ok 25 333325 generate-smoke-" in text and "failed" not in text
    assert world.ledger.path.read_bytes() == ledger
    writes = {name for name, _ in world.hub.calls[calls:]} & {"upload_file", "upload_folder", "create_repo"}
    assert not writes and world.submits() == [world.submits()[0]]

    world.script(billing_status=403)
    blind = flat(run("status"))
    assert "billing unavailable" in blind and "headroom 15600010 micro-USD" in blind
    world.script(billing_used_micro_usd=0)
    assert "billing 0 micro-USD ($0.00) used, 20000000 left of the credit" in flat(run("status"))

    # what a record carries is masked like any other output
    world.push_record(f"quant-full-{TOKEN}", notes=[])
    masked = run("status")
    assert "quant-full-hf_***" in masked.output and TOKEN not in masked.output
    world.script()
    assert run("wait", job_id).exit_code == 0
    assert "pending none" in flat(run("status"))


def test_status_says_so_when_the_run_records_cannot_be_read(world, monkeypatch):
    def unreachable(*args, **kwargs):
        raise LocalEntryNotFoundError("the Hub cannot be reached")

    with monkeypatch.context() as patch:
        patch.setattr(world.hub, "snapshot_download", unreachable)
        offline = run("status")
    del world.hub.repos[RESULTS]
    for result in (offline, run("status")):
        assert result.exit_code == 0 and "Traceback" not in result.output
        assert f"runs unavailable: {RESULTS} could not be read" in flat(result) and "ledger 0 micro-USD" in flat(result)


def test_five_verbs_in_one_process_read_the_token_once(world, monkeypatch):
    """Ready, plan, a dry run, table build and status read the token once and print it nowhere (TS-HUB-6)."""
    reads = []

    def getter():
        reads.append(1)
        return TOKEN

    evaluation(world, {"baseline": 0, "drifted": 2})
    fake = clients.make_jobs_client(world.settings)
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    monkeypatch.setattr(core_hub, "get_token", getter)
    # the real factory path, handing back the fake once it has read the token
    monkeypatch.setattr(cli.jobs_client, "register", lambda: clients.register_backend(lambda settings, api: fake))
    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", lambda settings, deps: Check(True, "fine")),))
    try:
        results = [
            run("ready"),
            run("plan", "drift"),
            run("launch", "drift", "--dry-run"),
            run("table", "build"),
            run("status"),
        ]
    finally:
        set_client_factory(default_client_factory)

    assert [result.exit_code for result in results] == [0, 0, 0, 0, 0]
    assert len(reads) == 1 and core_hub.token() == TOKEN
    local = [path.read_text() for path in (world.root / LOCAL).rglob("*") if path.is_file()]
    assert len(local) >= 3
    for text in (*(result.output for result in results), *local):
        assert TOKEN not in text
