import ast
import functools
import hashlib
import itertools
import json
import logging
import shutil
import string
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import typer
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils._http import default_client_factory
from hypothesis import HealthCheck, given
from hypothesis import settings as examples
from hypothesis import strategies as st
from typer.testing import CliRunner

from driftmap import cli, quant
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
    GenerationRow,
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
from driftmap.data import variants as data_variants
from driftmap.jobs import launcher, readiness
from driftmap.jobs.client import HfJobsClient
from driftmap.jobs.kinds import State
from driftmap.jobs.ledger import Ledger
from driftmap.jobs.plan import build_plan, pins
from driftmap.jobs.readiness import ITEMS, READY_PATH, Check, Item, ReadyReport
from driftmap.jobs.snapshot import DECISION_PATH, LOCAL, SMOKE_DIR, TABLES_PATH, Deps
from driftmap.judge import closed, reparse, rubric, validate
from driftmap.judge.incremental import dump_rows, judged_rows
from driftmap.repair import screen_gen, select
from driftmap.table import build as table_build
from driftmap.table import render
from driftmap.table.breakdown import Breakdown, Tags
from driftmap.table.breakdown import render as render_breakdown
from driftmap.table.build import NOT_COMPARABLE
from driftmap.table.variants import VariantTables
from driftmap.table.variants import render as render_variants
from tests.fakes import FakeHub, FakeJobsClient, judge_world, published
from tests.fakes import breakdown as fake_breakdown
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
VERBS = [
    "chain",
    "data",
    "decide",
    "figures",
    "judge",
    "launch",
    "ledger",
    "plan",
    "publish",
    "ready",
    "repair",
    "scan",
    "status",
    "table",
    "wait",
]
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
    assert groups == {
        "data": ["build", "variants"],
        "ledger": ["reconcile"],
        "table": ["breakdown", "build", "variants"],
        "judge": ["closed", "reparse", "validate"],
    }
    checks = typer.main.get_command(cli.app).commands["judge"].commands["validate"].commands
    assert sorted(checks) == ["retest", "run", "sample", "score"]
    scripts = typer.main.get_command(cli.app).commands["data"].commands["variants"].commands
    assert sorted(scripts) == ["accept", "build", "draft", "upload"]
    assert sorted(typer.main.get_command(cli.app).commands["decide"].commands) == ["drift"]
    assert sorted(typer.main.get_command(cli.app).commands["repair"].commands) == ["choose"]
    assert sorted(typer.main.get_command(cli.app).commands["publish"].commands) == ["check", "plan", "run"]


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
    for argv in (["plan", "repair"], ["launch", "repair"], ["chain", "steer"]):
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
        chosen=0.25 if status == "chosen" else None,
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


def reader(world, mode="full", target=None, **step):
    deps = Deps(client=None, hub=Hub(world.hub), ledger=world.ledger, repo_root=world.root)
    return cli._state(world.settings, deps, mode, target, **step)


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
    # a smoke run keeps its screen in its own tree
    world.push("smoke/alpha_screen.json", screen(settings, "chosen").model_dump_json().encode())
    world.push_record("repair-smoke-20261107T100000Z")
    assert reader(world, "smoke")("steer").recover_status == "chosen" and read("steer").recover_status is None
    # and its repair and quant runs push to repos of their own
    assert reader(world, "smoke")("quant").repaired_revision is None
    rehearsed = [Ref(kind="model", repo=f"{ref.repo}-smoke", revision=ref.revision) for ref in (repaired, gguf)]
    world.push_record("repair-smoke-20261107T100000Z", outputs=rehearsed[:1])
    world.push_record("quant-smoke-20261108T100000Z", outputs=rehearsed[1:])
    rehearsal = reader(world, "smoke")("quant_eval")
    assert (rehearsal.repaired_revision, rehearsal.gguf_record) == ("7" * 40, True)


def test_a_repair_launch_needs_its_drift_run_and_labels_and_an_apply_needs_a_chosen_screen(world, monkeypatch):
    # run ids are stamped to the second, and the screen and apply launches below are quicker than that
    seconds = itertools.count()
    stepping = functools.partial(Deps, run=git, now=lambda: datetime.now(UTC) + timedelta(seconds=next(seconds)))
    monkeypatch.setattr(cli, "Deps", stepping)
    dump(fake_labels.bare_tables(), world.root / TABLES_PATH)
    bare = run("launch", "repair", "--wait")
    assert bare.exit_code == 4 and bare.output.splitlines()[0].split() == ["verb", "launch"]
    missing = "drift record with merged_sha256; drifted benign labels for en, fr, es, ru, hi, ar"
    assert f"repair cannot run yet; missing: {missing}" in bare.output
    # a dry run says so too, and still prints the plan and the guard verdicts
    planned = run("launch", "repair", "--dry-run")
    assert planned.exit_code == 0 and "repair cannot run yet; missing: " in planned.output
    assert "estimate  399990 micro-USD" in planned.output and verdict(planned, "smoke") == "pass"

    world.hub.add_repo(EVAL, repo_type="dataset")
    repo, _, revision = SMOKE_DRIFTED.partition("@")
    merged = Ref(kind="model", repo=repo, revision=revision)
    # as the job reads it: the latest ok drift run of the mode, whatever config it ran under
    world.push_record("drift-smoke-20261105T090000Z", config_hash="0" * 64, merged_sha256="b" * 64, outputs=[merged])
    labels = fake_labels.column_rows("drifted", n=2, checkpoint=SMOKE_DRIFTED)
    put_labels(world, [row for row in labels if row.language != "ar"], "smoke")
    unlabelled = run("launch", "repair", "--wait")
    assert unlabelled.exit_code == 4 and "missing: drifted benign labels for ar\n" in unlabelled.output
    put_labels(world, labels, "smoke")

    applied = ["--", "--apply", "0.25"]
    early = run("launch", "repair", "--wait", *applied)
    assert early.exit_code == 4 and "missing: alpha screen with recover_status chosen" in early.output
    assert "`driftmap repair choose`" in early.output and world.submits() == []

    assert run("launch", "repair", "--wait").exit_code == 0
    (screened_run,) = world.submits()
    assert "driftmap.scripts.repair --run-id repair-smoke-" in screened_run and "--apply" not in screened_run
    # the screen run leaves its record undecided; only a chosen one lets an apply through
    for status in ("screened", "failed"):
        world.push("smoke/alpha_screen.json", screen(world.settings, status).model_dump_json().encode())
        for form in (applied, ["--", "--apply=0.25"], ["--", "--app", "0.25"]):
            assert run("launch", "repair", "--wait", *form).exit_code == 4, (status, form)
    world.push("smoke/alpha_screen.json", screen(world.settings, "chosen").model_dump_json().encode())
    # only the alpha the screen chose is pushed
    other = run("launch", "repair", "--wait", "--", "--apply", "0.9")
    assert other.exit_code == 4 and "missing: --apply 0.25: the screen chose that alpha, not 0.9" in other.output
    assert len(world.submits()) == 1
    pushed = run("launch", "repair", "--wait", *applied)
    assert pushed.exit_code == 0, pushed.output
    assert len(world.submits()) == 2 and world.submits()[-1].endswith(" --apply 0.25")


def test_job_arguments_are_read_as_the_job_will_before_anything_is_submitted(world, monkeypatch):
    with monkeypatch.context() as patch:
        # refused on the arguments alone
        patch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
        for kind, argv, said in (
            ("drift", ["--lr", "0.05"], "argument --lr: expected a rate above 0 and at most 0.01"),
            ("drift", ["--epochs", "0"], "argument --epochs: expected 1 or more"),
            ("drift", ["--lr"], "argument --lr: expected one argument"),
            ("drift", ["--learning-rate", "1e-4"], "unrecognized arguments: --learning-rate 1e-4"),
            ("drift", ["--preamble", "none"], "argument --preamble: invalid choice"),
            ("repair", ["--aply", "0.25"], "unrecognized arguments: --aply 0.25"),
            ("repair", ["--lr", "1e-4"], "unrecognized arguments: --lr 1e-4"),
            ("generate", [*BASELINE[1:], "--sets", "variants"], "--sets variants goes with --variants-rev SHA"),
            ("generate", [*BASELINE[1:], "--variants-rev", "c" * 40], "--sets variants goes with --variants-rev SHA"),
            ("generate", [*BASELINE[1:], "--sets", "variants", "--variants-rev", "main"], "the full 40-character"),
            ("generate", [*BASELINE[1:], "--sets", "harmful"], "argument --sets: invalid choice"),
            ("generate", [*BASELINE[1:], "--preamble", "none"], "argument --preamble: invalid choice"),
        ):
            refused = run("launch", kind, "--", *argv)
            assert refused.exit_code == 2 and said in flat(refused), argv
            assert f"the {kind} job would stop on its arguments" in flat(refused)

    dump(fake_labels.bare_tables(), world.root / TABLES_PATH)
    for argv, said in (
        (["--apply", "abc"], "--apply: expected one to 4 numbers above 0 and up to 1, got 'abc'"),
        (["--apply", "0.1,0.25"], "--apply takes one alpha, got '0.1,0.25'"),
        (["--apply", "0.25", "--alphas", "0.1"], "--apply pushes the chosen alpha and --alphas screens"),
        (["--alphas", "0.1,2"], "--alphas: expected one to 4 numbers"),
    ):
        refused = run("launch", "repair", "--", *argv)
        assert refused.exit_code == 2 and said in refused.stderr, argv
    assert world.submits() == [] and not world.ledger.path.exists()

    # what the job takes goes through, the runner's own --resume included
    resumed = ["--lr", "5e-4", "--epochs", "3", "--resume", "drift-smoke-20261105T090000Z"]
    planned = run("launch", "drift", "--dry-run", "--", *resumed)
    assert planned.exit_code == 0 and " --lr 5e-4 --epochs 3 --resume drift-smoke-20261105T090000Z\n" in planned.output
    assert run("launch", "repair", "--dry-run", "--", "--alphas", "0.1,0.25").exit_code == 0



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
            # screen, decide and apply are three launches with a local step between them
            (["repair"], "`driftmap repair choose`"),
            (["drift", "repair"], "driftmap launch repair --wait -- --apply ALPHA"),
        ):
            refused = run("chain", *argv)
            assert refused.exit_code == 2 and hint in flat(refused), argv

    blocked = run("chain", "generate", "steer", "--target", "drifted")
    assert blocked.exit_code == 4 and blocked.output.splitlines()[0].split() == ["verb", "chain"]
    assert "generate cannot run yet; missing: drift record with merged_sha256" in blocked.output
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


def put_labels(world, rows, mode="full", preamble="keep"):
    files = {}
    for row in rows:
        path = ids.labels_path(
            ids.slug(row.checkpoint), row.language, mode, benign=row.set == "benign", preamble=preamble
        )
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


def evaluation(world, columns, mode="full", n=4, listed=None, preamble="keep"):
    """Put prompt files, training rows, and per column labels and their generate run on the fake Hub."""
    if EVAL not in world.hub.repos:
        world.hub.add_repo(EVAL, repo_type="dataset")
    for column, unsafe in columns.items():
        # a smoke drift run pushes to its own repos
        changes = {"checkpoint": SMOKE_DRIFTED} if (column, mode) == ("drifted", "smoke") else {}
        put_labels(world, fake_labels.column_rows(column, n=n, unsafe=unsafe, **changes), mode, preamble)
        if preamble != "keep":
            run_id, notes = f"generate-{mode}-{column}-{preamble}", [f"preamble: {preamble}"]
            generated(world, column, changes.get("checkpoint"), mode=mode, run_id=run_id, notes=notes)
        else:
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
    first = run("table", "build")
    assert first.exit_code == 0 and drifted() == {fake_labels.CHECKPOINTS["drifted"]}
    assert unread(first) == [f"left unread: labels/{ids.slug(later)}: no ok full run on record read it"]
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
    # the guard reads the full table only: without one it waits, with one this flag does not reach it
    unbuilt = run("launch", "repair", "--dry-run")
    assert unbuilt.exit_code == 4 and "run `driftmap table build` first" in unbuilt.output
    dump(fake_labels.bare_tables(), world.root / TABLES_PATH)
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


HATE, PROFANITY, SELF_HARM = fake_breakdown.HATE, fake_breakdown.PROFANITY, fake_breakdown.SELF_HARM
# per language four harmful prompts: the first three carry hate speech, the third profanity too
TAGS = (((HATE,), "global"), ((HATE,), "global"), ((HATE, PROFANITY), "local"), ((SELF_HARM,), "local"))


def tagged(world, n=4):
    """Give the harmful prompts on the fake Hub their harm categories and origin."""
    for language in fake_labels.LANGUAGES:
        ids_ = fake_labels.prompt_ids("harmful", language, n)
        tags = {(language, prompt_id): Tags(*TAGS[index % len(TAGS)]) for index, prompt_id in enumerate(ids_)}
        rows = fake_breakdown.prompt_rows(tags, language)
        path = f"prompts/harmful/{language}.jsonl"
        world.hub.upload_file(path_or_fileobj=jsonl(rows), path_in_repo=path, repo_id=EVAL, repo_type="dataset")


def answered(world, columns, mode="full", preamble="keep", checkpoint=None):
    """Store the answers behind the harmful labels of ``evaluation``; per column, how many of each language were cut."""
    if RESPONSES not in world.hub.repos:
        world.hub.add_repo(RESPONSES, repo_type="dataset")
    for column, cut in columns.items():
        changes = {"checkpoint": SMOKE_DRIFTED} if (column, mode) == ("drifted", "smoke") else {}
        changes |= {"checkpoint": checkpoint} if checkpoint else {}
        labels = [row for row in fake_labels.column_rows(column, **changes) if row.set == "harmful"]
        for language in fake_labels.LANGUAGES:
            rows = [row for row in labels if row.language == language]
            answers = [fake_breakdown.answer(row, cut=index < cut) for index, row in enumerate(rows)]
            path = ids.responses_path(ids.slug(rows[0].checkpoint), "harmful", language, mode, preamble)
            data = judge_world.jsonl(answers)
            world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=RESPONSES, repo_type="dataset")


def breakdown_rate(found, condition, column, **level):
    wanted = {"harm_category": None, "language": None, "global_or_local": None, "length": None, **level}
    (row,) = [
        row
        for row in found.rates
        if (row.condition, row.column) == (condition, column)
        and all(getattr(row, name) == value for name, value in wanted.items())
    ]
    return row.k, row.n


def test_table_breakdown_reads_every_condition_writes_prints_and_uploads_without_a_readiness_report(world):
    evaluation(world, {"baseline": 2, "drifted": 1})
    evaluation(world, {"baseline": 3, "drifted": 0}, preamble="strip")
    answered(world, {"baseline": 2, "drifted": 1})
    answered(world, {"baseline": 1, "drifted": 0}, preamble="strip")
    tagged(world)
    (world.root / READY_PATH).unlink()
    stray = fake_labels.CHECKPOINTS["drifted"].replace("a" * 40, "e" * 40)
    put_labels(world, fake_labels.column_rows("drifted", unsafe=4, checkpoint=stray), preamble="strip")

    built = run("table", "breakdown")

    assert built.exit_code == 0, built.output
    assert built.output.splitlines()[:2] == ["verb      table breakdown", "mode      full"]
    local = world.root / TABLES_DIR / "breakdown.json"
    found = Breakdown.model_validate_json(local.read_bytes())
    assert found.conditions == {"keep": "served template", "strip": "no safety preamble"}
    assert found.columns == {"baseline": "baseline", "drifted": "fine-tuned"}
    hate = {"harm_category": HATE}
    assert breakdown_rate(found, "keep", "baseline", **hate) == (12, 18)
    assert breakdown_rate(found, "keep", "drifted", **hate) == (6, 18)
    assert breakdown_rate(found, "strip", "baseline", **hate) == (18, 18)
    assert breakdown_rate(found, "strip", "drifted") == (0, 24)
    assert breakdown_rate(found, "keep", "baseline", global_or_local="local") == (0, 12)
    # the stored answers say which were cut off at the token cap: here the two harmful ones of each language
    cut = {(row.condition, row.column): (row.k, row.n) for row in found.at_cap if row.language == "all"}
    assert cut == {
        ("keep", "baseline"): (12, 24),
        ("keep", "drifted"): (6, 24),
        ("strip", "baseline"): (6, 24),
        ("strip", "drifted"): (0, 24),
    }
    assert breakdown_rate(found, "keep", "baseline", length="truncated") == (12, 12)
    assert breakdown_rate(found, "keep", "baseline", length="complete") == (0, 12)
    assert "### Answers cut off at the cap of 256 new tokens" in built.output
    # their text goes nowhere
    assert "placeholder response" not in built.output and b"placeholder response" not in local.read_bytes()
    # labels no finished run generated are left on the Hub, under either condition
    assert unread(built) == [f"left unread: nopreamble/labels/{ids.slug(stray)}: no ok full run on record read it"]
    shown = render_breakdown(found)
    assert shown in built.output and "## No safety preamble" in shown
    assert built.output.index("# Harmful-response rate") < built.output.index("written to .run-local/results/tables")
    assert (world.root / TABLES_DIR / "breakdown.md").read_text(encoding="utf-8") == shown
    assert stored(world, "tables/breakdown.json") == local.read_bytes()
    assert stored(world, "tables/breakdown.md") == shown.encode()
    assert built.output.splitlines()[-1] == f"uploaded to {RESULTS} under tables/"
    # the result tables and the drift flag are not its to write
    assert not (world.root / TABLES_PATH).exists()
    assert sorted(path for path in world.hub.repos[RESULTS]["main"] if "tables/" in path) == [
        "tables/breakdown.json",
        "tables/breakdown.md",
    ]


def test_table_breakdown_covers_the_safety_patch_and_reads_nothing_of_the_script_variants(world):
    evaluation(world, {"baseline": 2, "drifted": 1})
    evaluation(world, {"baseline": 1, "drifted": 0}, preamble="patch")
    answered(world, {"baseline": 0, "drifted": 0})
    answered(world, {"baseline": 0, "drifted": 0}, preamble="patch")
    tagged(world)
    # later runs over the variants read another revision of the fine-tuned checkpoint and left labels of their own
    other = fake_labels.CHECKPOINTS["drifted"].replace("a" * 40, "9" * 40)
    generated(world, "drifted", other, run_id="generate-full-drifted-variants", notes=["sets: variants"])
    both = ["preamble: patch", "sets: variants"]
    generated(world, "drifted", other, run_id="generate-full-drifted-patch-variants", notes=both)
    rows = [fake_labels.label("drifted", "hi-Latn", index, checkpoint=other, harm="yes") for index in range(4)]
    data = "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()
    for preamble in ("keep", "patch"):
        path = ids.labels_path(ids.slug(other), "hi-Latn", "full", preamble=preamble, variants=True)
        world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")

    built = run("table", "breakdown")

    assert built.exit_code == 0, built.output
    found = Breakdown.model_validate_json((world.root / TABLES_DIR / "breakdown.json").read_bytes())
    assert found.conditions == {"keep": "served template", "patch": "safety patch"}
    rates = {name: breakdown_rate(found, *name) for name in itertools.product(("keep", "patch"), found.columns)}
    assert rates == {
        ("keep", "baseline"): (12, 24),
        ("keep", "drifted"): (6, 24),
        ("patch", "baseline"): (6, 24),
        ("patch", "drifted"): (0, 24),
    }
    shown = render_breakdown(found)
    assert shown in built.output and "## Safety patch" in shown and "## No safety preamble" not in shown
    assert "left unread" not in built.output and "hi-Latn" not in built.output


def test_table_breakdown_leaves_out_a_column_its_generate_run_marks_not_comparable(world):
    evaluation(world, {"baseline": 2, "drifted": 1})
    answered(world, {"baseline": 2, "drifted": 1})
    tagged(world)
    generated(world, "drifted", notes=[f"{NOT_COMPARABLE}another seed"])

    built = run("table", "breakdown")

    assert built.exit_code == 0, built.output
    found = Breakdown.model_validate_json((world.root / TABLES_DIR / "breakdown.json").read_bytes())
    assert found.columns == {"baseline": "baseline"} and found.changes == [] and found.transitions == []
    assert [(row.condition, row.column, row.state, row.reason) for row in found.not_tabled] == [
        ("keep", "drifted", "not_comparable", "another seed")
    ]
    assert "- fine-tuned, served template: not comparable: another seed" in built.output


def test_table_breakdown_refuses_prompts_without_tags_and_missing_labels(world):
    evaluation(world, {})
    tagged(world)
    unjudged = run("table", "breakdown")
    assert unjudged.exit_code == 2 and "no judged labels of harmful prompts" in unjudged.stderr
    evaluation(world, {"baseline": 2, "drifted": 1})
    unstored = run("table", "breakdown")
    assert unstored.exit_code == 1 and f"{RESPONSES} could not be read" in unstored.stderr
    answered(world, {"baseline": 2})
    untagged = run("table", "breakdown")
    assert untagged.exit_code == ExitCode.INTEGRITY and "harmful/ar.jsonl:1: not a row" in untagged.stderr
    tagged(world, n=5)
    short = run("table", "breakdown")
    assert short.exit_code == ExitCode.INTEGRITY and "baseline harmful/ar: 4 rows for 5 prompts" in short.stderr
    # a label needs the answer it was made from
    tagged(world)
    unanswered = run("table", "breakdown")
    assert unanswered.exit_code == ExitCode.INTEGRITY
    assert "keep/drifted: no stored answer to harmful/ar/harmful-ar-000" in unanswered.stderr
    for result in (unjudged, unstored, untagged, short, unanswered):
        assert "Traceback" not in result.output and not (world.root / TABLES_DIR / "breakdown.json").exists()
    assert not [path for path in world.hub.repos[RESULTS]["main"] if "breakdown" in path]


def test_a_smoke_breakdown_keeps_to_its_own_tree_and_one_that_cannot_be_uploaded_stays_local(world, monkeypatch):
    # six prompts listed, the smoke runs took the first four
    evaluation(world, {"baseline": 2, "drifted": 0}, mode="smoke")
    answered(world, {"baseline": 3, "drifted": 0}, mode="smoke")
    tagged(world, n=6)
    assert "no judged labels" in run("table", "breakdown").stderr

    built = run("table", "breakdown", "--mode", "smoke")

    assert built.exit_code == 0, built.output
    local = world.root / LOCAL / "results" / "smoke" / "tables"
    found = Breakdown.model_validate_json((local / "breakdown.json").read_bytes())
    assert breakdown_rate(found, "keep", "baseline") == (12, 24)
    assert [(row.column, row.k, row.n) for row in found.at_cap if row.language == "all"] == [
        ("baseline", 18, 24),
        ("drifted", 0, 24),
    ]
    # one condition alone is broken down as it is
    assert found.conditions == {"keep": "served template"}
    assert stored(world, "smoke/tables/breakdown.json") == (local / "breakdown.json").read_bytes()
    assert stored(world, "smoke/tables/breakdown.md") == (local / "breakdown.md").read_bytes()
    assert not (world.root / TABLES_DIR).exists() and "tables/breakdown.json" not in world.hub.repos[RESULTS]["main"]

    (local / "breakdown.json").unlink()
    monkeypatch.setattr(world.hub, "upload_folder", lambda **call: world.hub._raise(403, "https://huggingface.co/x"))
    kept = run("table", "breakdown", "--mode", "smoke")
    assert kept.exit_code == ExitCode.PUSH_FAILED and "run the command again" in kept.stderr
    assert "breakdown.json and breakdown.md are in .run-local/results/smoke/tables" in kept.stderr
    assert (local / "breakdown.json").exists()


def variant_prompts(world, code, count):
    """Put ``count`` variant prompts of the harmful set on the fake Hub, tied to their parents; the commit they make."""
    parent = code.partition("-")[0]
    rows = [
        {
            "prompt_id": f"variant-{code}-{index:03d}",
            "parent_prompt_id": fake_labels.prompt_ids("harmful", parent, index + 1)[-1],
            "parent_language": parent,
            "language": code,
            "set": "harmful",
            "prompt": f"placeholder variant prompt {index:03d}",
        }
        for index in range(count)
    ]
    path = ids.variant_prompts("harmful", code)
    return world.hub.upload_file(path_or_fileobj=jsonl(rows), path_in_repo=path, repo_id=EVAL, repo_type="dataset").oid


def variants_run(world, column, commit, harmful, code="hi-Latn", mode="full", preamble="keep", **changed):
    """Record a run of ``column`` over the variants at ``commit``; store its four labels, ``harmful`` of them so."""
    checkpoint = fake_labels.CHECKPOINTS[column]
    rows = [
        fake_labels.label(
            column,
            code,
            index,
            harm="yes" if index < harmful else "no",
            **{"prompt_id": f"variant-{code}-{index:03d}", **changed.get(str(index), {})},
        )
        for index in range(4 if harmful is not None else 0)
    ]
    data = "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()
    path = ids.labels_path(ids.slug(checkpoint), code, mode, preamble=preamble, variants=True)
    if rows:
        world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")
    notes = [*([f"preamble: {preamble}"] if preamble != "keep" else []), "sets: variants"]
    inputs = [Ref(kind="dataset", repo=EVAL, revision=revision) for revision in ("6" * 40, "6" * 40, commit)]
    run_id = f"generate-{mode}-{column}-{preamble}-variants"
    return generated(world, column, run_id=run_id, mode=mode, notes=notes, inputs=inputs)


def test_table_variants_pairs_each_columns_variants_with_their_parents_writes_prints_and_uploads(world):
    evaluation(world, {"baseline": 1, "drifted": 0})
    evaluation(world, {"baseline": 0}, preamble="patch")
    first = variant_prompts(world, "hi-Latn", 3)
    variants_run(world, "baseline", first, 3)
    # the variants grew before the fine-tuned model read them; its fourth rewrite reads as another request
    second = variant_prompts(world, "hi-Latn", 4)
    variants_run(world, "drifted", second, 2, **{"3": {"request_harm": "no"}})
    variants_run(world, "baseline", second, 1, preamble="patch")
    (world.root / READY_PATH).unlink()

    built = run("table", "variants")

    assert built.exit_code == 0, built.output
    assert built.output.splitlines()[:2] == ["verb      table variants", "mode      full"]
    local = world.root / TABLES_DIR / "variants.json"
    found = VariantTables.model_validate_json(local.read_bytes())
    assert found.conditions == {"keep": "served template", "patch": "safety patch"}
    assert found.columns == {"baseline": "baseline", "drifted": "fine-tuned"}
    sources = {(source.condition, source.column): source for source in found.sources}
    assert {key: (source.variants_revision, source.judged, source.left_out) for key, source in sources.items()} == {
        ("keep", "baseline"): (first, 4, 0),
        ("keep", "drifted"): (second, 4, 1),
        ("patch", "baseline"): (second, 4, 0),
    }
    assert sources["keep", "drifted"].checkpoint == fake_labels.CHECKPOINTS["drifted"]
    rows = {(row.condition, row.column, row.code): row for row in found.rows}
    assert sorted(rows) == [(*key, code) for key in sorted(sources) for code in ("all-Latn", "hi-Latn")]
    # a label of a variant the run's commit does not hold is not paired
    base = rows["keep", "baseline", "hi-Latn"]
    assert (base.n_pairs, base.parent, base.variant, base.delta) == pytest.approx((3, 1 / 3, 1.0, 2 / 3))
    tuned = rows["keep", "drifted", "hi-Latn"]
    assert (tuned.n_pairs, tuned.parent, tuned.variant) == pytest.approx((3, 0.0, 2 / 3))
    patched = rows["patch", "baseline", "all-Latn"]
    assert (patched.n_pairs, patched.parent, patched.variant) == pytest.approx((4, 0.0, 0.25))
    shown = render_variants(found)
    assert shown in built.output and "### Fine-tuned" in shown and "## Safety patch" in shown
    assert built.output.index("# Script variants") < built.output.index("written to .run-local/results/tables")
    assert (world.root / TABLES_DIR / "variants.md").read_text(encoding="utf-8") == shown
    assert stored(world, "tables/variants.json") == local.read_bytes()
    assert stored(world, "tables/variants.md") == shown.encode()
    assert built.output.splitlines()[-1] == f"uploaded to {RESULTS} under tables/"
    # the result tables and the drift flag are not its to write
    assert not (world.root / TABLES_PATH).exists()
    assert sorted(path for path in world.hub.repos[RESULTS]["main"] if "tables/" in path) == [
        "tables/variants.json",
        "tables/variants.md",
    ]


def test_table_variants_refuses_without_a_variants_run_its_labels_or_its_prompts(world):
    evaluation(world, {"baseline": 1})
    unrun = run("table", "variants")
    assert unrun.exit_code == 2 and "no finished full run over the script variants is on record" in unrun.stderr
    commit = variant_prompts(world, "hi-Latn", 4)
    variants_run(world, "baseline", commit, None)
    unjudged = run("table", "variants")
    assert unjudged.exit_code == 2 and "no judged labels of script variants" in unjudged.stderr
    record = variants_run(world, "baseline", commit, 2)
    assert run("table", "variants", "--mode", "smoke").exit_code == 2
    record.inputs[-1] = Ref(kind="dataset", repo=EVAL, revision="e" * 40)
    world.push(f"runs/{record.run_id}.json", record.model_dump_json(by_alias=True).encode())
    gone = run("table", "variants")
    assert gone.exit_code == 1 and f"{EVAL} could not be read" in gone.stderr
    untied = world.hub.upload_file(
        path_or_fileobj=jsonl([{"prompt_id": "variant-hi-Latn-000", "language": "hi-Latn"}]),
        path_in_repo=ids.variant_prompts("harmful", "hi-Latn"),
        repo_id=EVAL,
        repo_type="dataset",
    ).oid
    record.inputs[-1] = Ref(kind="dataset", repo=EVAL, revision=untied)
    world.push(f"runs/{record.run_id}.json", record.model_dump_json(by_alias=True).encode())
    torn = run("table", "variants")
    assert torn.exit_code == ExitCode.INTEGRITY and "harmful/hi-Latn.jsonl:1: not a row" in torn.stderr
    for result in (unrun, unjudged, gone, torn):
        assert "Traceback" not in result.output and not (world.root / TABLES_DIR / "variants.json").exists()
    assert not [path for path in world.hub.repos[RESULTS]["main"] if "variants" in path and "tables" in path]


def test_a_smoke_variants_table_keeps_to_its_own_tree_and_one_that_cannot_be_uploaded_stays_local(world, monkeypatch):
    evaluation(world, {"baseline": 2}, mode="smoke")
    variants_run(world, "baseline", variant_prompts(world, "es-mixed", 4), 1, code="es-mixed", mode="smoke")
    assert "no finished full run over the script variants" in run("table", "variants").stderr

    built = run("table", "variants", "--mode", "smoke")

    assert built.exit_code == 0, built.output
    local = world.root / LOCAL / "results" / "smoke" / "tables"
    found = VariantTables.model_validate_json((local / "variants.json").read_bytes())
    assert [(row.code, row.rewrite, row.n_pairs) for row in found.rows] == [
        ("es-mixed", "codemixed", 4),
        ("all-mixed", "codemixed", 4),
    ]
    assert stored(world, "smoke/tables/variants.json") == (local / "variants.json").read_bytes()
    assert stored(world, "smoke/tables/variants.md") == (local / "variants.md").read_bytes()
    assert not (world.root / TABLES_DIR).exists() and "tables/variants.json" not in world.hub.repos[RESULTS]["main"]

    monkeypatch.setattr(world.hub, "upload_folder", lambda **call: world.hub._raise(403, "https://huggingface.co/x"))
    kept = run("table", "variants", "--mode", "smoke")
    assert kept.exit_code == ExitCode.PUSH_FAILED and "run the command again" in kept.stderr
    assert "variants.json and variants.md are in .run-local/results/smoke/tables" in kept.stderr


# the script variants and the safety patch have no data in the stored files
FIGURE_STEMS = (
    "f1_harmful_by_language",
    "f2_category_by_language",
    "f3_three_axes",
    "f4_harm_categories",
    "f5_transitions",
    "f6_truncation",
)
FIGURES = [f"{stem}.{kind}" for stem in FIGURE_STEMS for kind in ("png", "svg")]


def test_figures_are_drawn_from_the_stored_tables_and_breakdown_without_the_hub(world, tmp_path, monkeypatch):
    folder = world.root / TABLES_DIR
    calls = len(world.hub.calls)
    absent = run("figures")
    assert absent.exit_code == 2 and "holds no tables.json; run `driftmap table build` first" in absent.stderr
    tables = published.two_column_tables(tmp_path, world.settings)
    dump(tables, folder / "tables.json")
    unbroken = run("figures")
    assert (
        unbroken.exit_code == 2 and "holds no breakdown.json; run `driftmap table breakdown` first" in unbroken.stderr
    )
    dump(fake_breakdown.two_conditions(tmp_path, world.settings), folder / "breakdown.json")
    (world.root / READY_PATH).unlink()

    drawn = run("figures")

    assert drawn.exit_code == 0, drawn.output
    assert drawn.output.splitlines()[:2] == ["verb      figures", "out       docs/figures"]
    assert drawn.output.splitlines()[-len(FIGURES) :] == [f"docs/figures/{name}" for name in FIGURES]
    # a figure without its data is not drawn, and said so
    assert drawn.output.splitlines()[-len(FIGURES) - 2 : -len(FIGURES)] == [
        "not drawn: f7_script_variants: needs variants.json beside the tables, holding the pooled harmful pairings",
        "not drawn: f8_safety_patch: needs the safety-patch condition in the breakdown",
    ]
    assert sorted(path.name for path in (world.root / "docs" / "figures").iterdir()) == sorted(FIGURES)
    assert all((world.root / "docs" / "figures" / name).stat().st_size > 2000 for name in FIGURES)
    assert len(world.hub.calls) == calls and not world.client_log.exists()

    # the second condition's tables and the stored script-variant pairings join when they are there
    dump(tables, folder / "tables-nopreamble.json")
    pairing = {"set": "harmful_romanized", "language": "all", "metric": "harmful", "n_pairs": 300, "parent": 0.1}
    pairing |= {"variant": 0.25, "delta": 0.15, "ci": [0.09, 0.21], "p_value": 0.0001}
    (folder / "variants.json").write_text(json.dumps({"baseline": [pairing], "drifted": [pairing]}))
    elsewhere = run("figures", "--out", "elsewhere")
    assert elsewhere.exit_code == 0, elsewhere.output
    assert elsewhere.output.splitlines()[-2:] == [f"elsewhere/f7_script_variants.{kind}" for kind in ("png", "svg")]
    assert [line for line in elsewhere.output.splitlines() if line.startswith("not drawn: ")] == [
        "not drawn: f8_safety_patch: needs the safety-patch condition in the breakdown"
    ]
    folders = (world.root / "docs" / "figures", world.root / "elsewhere")
    one, both = ([(where / name).read_bytes() for name in FIGURES] for where in folders)
    # the figure of the tables changes, the one of the breakdown is the same file again
    assert one[0] != both[0] and one[2] == both[2]
    # and so do the tables of the runs under the safety patch
    dump(tables, folder / "tables-patched.json")
    third = run("figures", "--out", "third")
    assert third.exit_code == 0, third.output
    assert (world.root / "third" / FIGURES[0]).read_bytes() not in (one[0], both[0])

    # the other fine-tune's figures come from its own files and go to a folder of their own
    unbuilt = run("figures", "--fine-tune", "aggressive")
    assert unbuilt.exit_code == 2
    assert "holds no tables-aggressive.json; run `driftmap table build --fine-tune aggressive` first" in unbuilt.stderr
    dump(tables, folder / "tables-nopreamble-aggressive.json")
    dump(fake_breakdown.two_conditions(tmp_path / "other", world.settings), folder / "breakdown-aggressive.json")
    apart = run("figures", "--fine-tune", "aggressive")
    assert apart.exit_code == 0, apart.output
    assert apart.output.splitlines()[1:3] == ["out       docs/figures-aggressive", "fine-tune aggressive"]
    # the stored pairings are not of this fine-tune
    assert sorted(path.name for path in (world.root / "docs" / "figures-aggressive").iterdir()) == sorted(FIGURES)
    assert [(where / FIGURES[0]).read_bytes() for where in folders] == [one[0], both[0]]

    (folder / "variants.json").write_text(json.dumps({"baseline": [{"set": "harmful_romanized"}]}))
    unpaired = run("figures")
    assert unpaired.exit_code == 2 and "variants.json: not the pairings of each checkpoint" in unpaired.stderr
    (folder / "variants.json").unlink()
    (folder / "breakdown.json").write_text("{")
    torn = run("figures")
    assert torn.exit_code == 2 and "breakdown.json: " in torn.stderr
    monkeypatch.delattr("driftmap.figures.draw", raising=False)
    monkeypatch.setitem(sys.modules, "driftmap.figures.draw", None)
    bare = run("figures")
    assert bare.exit_code == 2 and "uv sync --extra figures" in bare.stderr and "Traceback" not in bare.output


GGUF = f"{fake_labels.CHECKPOINTS['repaired_q4']}#{quant.GGUF_FILE}"
STEERED = fake_labels.CHECKPOINTS["drifted"] + "--steer-l24-s1.0"


def quant_evaluated(world, checkpoint=GGUF, **changes):
    """Record a quant-eval run as the job writes it: the GGUF first, then the checkpoint it was converted from."""
    record = generated(world, "repaired_q4", checkpoint.partition("#")[0], **{"kind": "quant_eval", **changes})
    source, revision = fake_labels.REFS["repaired"]
    record.inputs.append(Ref(kind="model", repo=source + ids.repo_suffix(record.mode), revision=revision))
    world.push(f"runs/{record.run_id}.json", record.model_dump_json(by_alias=True).encode())


def states(world):
    tables = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    return {column: (state.state, state.kind) for column, state in tables.columns.items()}


def unread(result):
    return [line for line in result.output.splitlines() if line.startswith("left unread: ")]


def test_the_q4_column_follows_quant_eval_runs_and_names_the_folders_it_leaves(world):
    evaluation(world, {"baseline": 0, "drifted": 2, "repaired": 1})
    put_labels(world, fake_labels.column_rows("repaired_q4", unsafe=1, checkpoint=GGUF))
    slug = ids.slug(GGUF)

    # labels alone do not fill the column: no finished run says which file they belong to
    bare = run("table", "build")
    assert bare.exit_code == 0 and states(world)["repaired_q4"] == ("not_run", "q4_k_m")
    assert unread(bare) == [f"left unread: labels/{slug}: no ok full run on record read it"]
    for changes in ({"status": "failed"}, {"mode": "smoke"}, {"kind": "quant"}):
        quant_evaluated(world, run_id="quant_eval-full-1", **changes)
        assert run("table", "build").exit_code == 0 and states(world)["repaired_q4"][0] != "tabled", changes

    quant_evaluated(world, run_id="quant_eval-full-1")
    built = run("table", "build")

    assert built.exit_code == 0, built.output
    assert states(world)["repaired_q4"] == ("tabled", "q4_k_m") and unread(built) == []
    tables = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    assert GGUF in {row.checkpoint for row in tables.safety}

    # a later file takes the column; the earlier folder stays on the Hub, named
    later = GGUF.replace("c" * 40, "e" * 40)
    put_labels(world, fake_labels.column_rows("repaired_q4", checkpoint=later))
    put_labels(world, fake_labels.column_rows("baseline", checkpoint=f"{OWNER}/some-other-model@{'9' * 40}"))
    quant_evaluated(world, later, run_id="quant_eval-full-2", started=datetime(2026, 11, 14, tzinfo=UTC))
    moved = run("table", "build")
    assert moved.exit_code == 0, moved.output
    tables = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    assert {row.checkpoint for row in tables.safety if "GGUF" in row.checkpoint} == {later}
    assert unread(moved) == [
        "left unread: labels/some-other-model--999999999999: not a checkpoint of this table",
        f"left unread: labels/{slug}: a later ok run read {ids.slug(later)}",
    ]


def test_a_smoke_table_follows_the_smoke_quant_eval_run(world):
    evaluation(world, {"baseline": 0}, mode="smoke")
    rehearsed = GGUF.replace("@", "-smoke@")
    put_labels(world, fake_labels.column_rows("repaired_q4", checkpoint=rehearsed), mode="smoke")
    quant_evaluated(world, rehearsed, mode="smoke", run_id="quant_eval-smoke-1")

    built = run("table", "build", "--mode", "smoke")

    assert built.exit_code == 0, built.output
    local = world.root / LOCAL / "results" / "smoke" / "tables" / "tables.json"
    tables = Tables.model_validate_json(local.read_bytes())
    assert tables.columns["repaired_q4"].state == "tabled" and unread(built) == []
    assert rehearsed in {row.checkpoint for row in tables.safety}


def test_a_steered_column_follows_steer_runs(world):
    evaluation(world, {"baseline": 0, "drifted": 2})
    steered = [row.model_copy(update={"checkpoint": STEERED}) for row in fake_labels.column_rows("drifted")]
    put_labels(world, steered)
    folder = f"labels/{ids.slug(STEERED)}"

    bare = run("table", "build")
    assert bare.exit_code == 0 and states(world)["repaired"] == ("not_run", "weights")
    assert unread(bare) == [f"left unread: {folder}: no ok full run on record read it"]

    # the steer run records the weights it steered; its folder is theirs with the steer suffix
    generated(world, "drifted", kind="steer", run_id="steer-full-1")
    built = run("table", "build")

    assert built.exit_code == 0, built.output
    assert states(world) == {
        "baseline": ("tabled", "weights"),
        "drifted": ("tabled", "weights"),
        "repaired": ("tabled", "inference-time"),
        "repaired_q4": ("not_run", "q4_k_m"),
    }
    assert unread(built) == []

    # a steer run on other weights does not speak for this folder
    other = fake_labels.CHECKPOINTS["drifted"].replace("a" * 40, "e" * 40)
    generated(world, "drifted", other, kind="steer", run_id="steer-full-1")
    moved = run("table", "build")
    assert moved.exit_code == 0 and states(world)["repaired"] == ("not_run", "weights")
    assert unread(moved) == [f"left unread: {folder}: no ok full run on record read it"]


def test_a_deleted_local_table_does_not_open_the_drift_gate(world):
    evaluation(world, {"baseline": 0, "drifted": 0})
    assert run("table", "build").exit_code == 8
    (world.root / TABLES_PATH).unlink()

    refused = run("launch", "repair", "--dry-run")

    assert refused.exit_code == 4 and verdict(refused, "drift_decision") == "refused"
    assert "run `driftmap table build` first" in refused.output
    assert verdict(run("launch", "generate", "--dry-run", *BASELINE), "drift_decision") == "pass"
    # the build brings the flagged copy back from the Hub, and the decision is asked for again
    assert run("table", "build").exit_code == 0
    again = run("launch", "repair", "--dry-run")
    assert again.exit_code == 4 and "driftmap decide drift" in again.output


@pytest.mark.parametrize("lose", [lambda path: path.write_text("{"), Path.unlink], ids=["torn", "missing"])
def test_a_torn_or_missing_local_table_takes_its_drift_flag_from_the_copy_on_the_hub(world, lose):
    evaluation(world, {"baseline": 0, "drifted": 0})
    assert run("table", "build").exit_code == 8
    local, on_hub = world.root / TABLES_PATH, stored(world, "tables/tables.json")

    # the flag the Hub copy holds is on file, not new
    lose(local)
    assert run("table", "build").exit_code == 0
    assert Tables.model_validate_json(local.read_bytes()).flags["drift_below_threshold"] is True

    lose(local)
    generated(world, "drifted", notes=[f"{NOT_COMPARABLE}another seed"])
    again = run("table", "build")

    # the flag can no longer be evaluated: it stays, locally and on the Hub, and so does the guard
    assert again.exit_code == ExitCode.DRIFT_BELOW_THRESHOLD and "driftmap decide drift" in again.stderr
    assert local.read_bytes() == on_hub == stored(world, "tables/tables.json")
    assert run("launch", "repair", "--dry-run").exit_code == 4

    # a Hub copy that does not parse either holds no flag
    lose(local)
    world.push("tables/tables.json", b"{")
    assert run("table", "build").exit_code == 0
    assert "drift_below_threshold" not in json.loads(stored(world, "tables/tables.json"))["flags"]


LINK = "https://huggingface.co/blog/vincentnkq/drift-map"
PUBLISHED = (*published.DATASETS, *published.MODELS)


@pytest.fixture
def shelf(monkeypatch):
    """A checkout whose Hub holds the project's repos as the jobs leave them: private, uncarded, unlisted."""
    fake = FakeHub(Path.cwd().with_name("published-hub"))
    published.seed(fake)
    return World(monkeypatch, Path.cwd(), hub=fake)


def test_publish_plan_prints_the_plan_and_exits_with_its_first_problem(shelf):
    # nothing is launched, so a stale readiness report is no obstacle
    (shelf.root / READY_PATH).unlink()

    clean = run("publish", "plan")

    assert clean.exit_code == 0, clean.output
    assert clean.output.splitlines()[0].split() == ["verb", "publish", "plan"]
    assert "history scan: 21 file fetches over 6 commits" in clean.output
    eval_line = f"{published.EVAL} (dataset): private -> public; card: write; delete 2 files; collection: add"
    assert eval_line in clean.output and f"stays private: {published.MERGED} (model)" in clean.output
    assert "problem: " not in clean.output

    shelf.hub._commit(published.RESULTS, "job upload", {"logs/judge.log": f"token {TOKEN}\n".encode()})
    refused = run("publish", "plan", "--writeup", LINK)

    assert refused.exit_code == ExitCode.REDACTION == 6
    assert f"writeup   {LINK}" in refused.output
    assert [line.rpartition("/logs/")[2] for line in refused.output.splitlines() if line.startswith("problem: ")] == [
        "judge.log:1: hf_token"
    ]
    assert "rotate the secret" in refused.output and TOKEN not in refused.output
    assert "1 problem: `driftmap publish run` would refuse" in refused.stderr
    assert published.writes(shelf.hub) == [] and set(PUBLISHED) <= shelf.hub.private


def test_a_publish_dry_run_prints_the_plan_and_writes_nothing(shelf):
    dry = run("publish", "run", "--dry-run", "--delete-intermediates", "--writeup", LINK)

    assert dry.exit_code == 0, dry.output
    assert dry.output.splitlines()[:4] == [
        "verb      publish run",
        f"writeup   {LINK}",
        "dry-run   yes",
        "delete    yes",
    ]
    assert f"{published.GGUF} (model): private -> public; card: write; " in dry.output
    assert f"would delete: {published.MERGED} (model)" in dry.output
    assert dry.output.splitlines()[-1] == "dry run: nothing was written"

    # a dry run exits as the run would
    shelf.hub._commit(published.RESULTS, "job upload", {"logs/judge.log": f"token {TOKEN}\n".encode()})
    refused = run("publish", "run", "--dry-run")
    assert refused.exit_code == ExitCode.REDACTION == 6
    assert "logs/judge.log:1: hf_token" in refused.stderr and TOKEN not in refused.output
    assert refused.stdout.splitlines()[-1] == "dry run: nothing was written"
    assert published.writes(shelf.hub) == [] and set(PUBLISHED) <= shelf.hub.private
    assert {published.MERGED, *published.SMOKE} <= set(shelf.hub.repos)


def test_publish_run_cards_opens_and_lists_and_the_check_agrees(shelf):
    before = run("publish", "check")
    assert before.exit_code == ExitCode.INTEGRITY == 12 and before.output.splitlines()[0] == "verb      publish check"
    assert f"- {published.EVAL} is private" in before.output and published.writes(shelf.hub) == []

    done = run("publish", "run", "--writeup", LINK)

    assert done.exit_code == 0, done.output
    assert not (shelf.hub.private | shelf.hub.gated) & set(PUBLISHED)
    assert {published.MERGED, *published.SMOKE} <= shelf.hub.private
    for repo in PUBLISHED:
        assert f"- Write-up: {LINK}" in shelf.hub.repos[repo]["main"]["README.md"].decode()
        assert f"card: {repo}\n" in done.output and f"listed: {repo} in " in done.output
    assert f"public: {published.EVAL} (dataset)" in done.output

    shelf.hub.calls.clear()
    after = run("publish", "check")
    assert after.exit_code == 0 and published.writes(shelf.hub) == []
    assert f"{published.EVAL} yes yes no yes yes apache-2.0, cc-by-4.0" in flat(after)
    assert f"{published.MERGED} yes no no no no -" in flat(after)
    again = run("publish", "run", "--writeup", LINK)
    assert again.exit_code == 0 and f"already public: {published.GGUF} (model)" in again.output


def test_a_failed_publish_exits_7_and_names_the_command_to_run_again(shelf):
    real, left = shelf.hub.upload_folder, [1]

    def upload_folder(**kwargs):
        if kwargs["repo_id"] == published.RESULTS and left[0]:
            left[0] -= 1
            shelf.hub._raise(403, f"https://huggingface.co/api/datasets/{published.RESULTS}/commit/main")
        return real(**kwargs)

    shelf.hub.upload_folder = upload_folder

    failed = run("publish", "run", "--delete-intermediates", "--writeup", LINK)

    assert failed.exit_code == ExitCode.PUSH_FAILED == 7
    assert failed.stderr.splitlines() == [
        f"error: the card upload to {published.RESULTS} failed (403); public now: none",
        f"resume: driftmap publish run --writeup {LINK} --delete-intermediates; finished steps are skipped",
    ]
    assert "deleted: " + published.MERGED in failed.output and set(PUBLISHED) <= shelf.hub.private
    resumed = run("publish", "run", "--writeup", LINK, "--delete-intermediates")
    assert resumed.exit_code == 0, resumed.output
    assert not shelf.hub.private & set(PUBLISHED) and published.MERGED not in shelf.hub.repos


def test_publish_run_fresh_opens_a_clean_copy_of_a_repo_whose_history_holds_a_token(shelf):
    shelf.hub._commit(published.RESULTS, "job upload", {"logs/judge.log": f"token {TOKEN}\n".encode()})
    shelf.hub._commit(published.RESULTS, "remove it", None, ["logs/judge.log"])
    work = f"{published.RESULTS}-work"

    refused = run("publish", "run")
    assert refused.exit_code == ExitCode.REDACTION and "publish with --fresh" in refused.stderr
    assert published.writes(shelf.hub) == []

    real = shelf.hub.create_repo
    shelf.hub.create_repo = lambda repo_id, **kwargs: shelf.hub._raise(409, "https://huggingface.co/api/repos/create")
    failed = run("publish", "run", "--fresh")
    assert failed.exit_code == ExitCode.PUSH_FAILED and "fresh     yes" in failed.output
    assert failed.stderr.splitlines() == [
        f"error: the clean copy of {published.RESULTS} failed (409); public now: none",
        "resume: driftmap publish run --fresh; finished steps are skipped",
    ]
    assert work in shelf.hub.private and published.RESULTS not in shelf.hub.repos

    shelf.hub.create_repo = real
    done = run("publish", "run", "--fresh")
    assert done.exit_code == 0, done.output
    assert f"clean copy: {published.RESULTS} (dataset) from {work}@" in done.output
    assert work in shelf.hub.private and not shelf.hub.private & set(PUBLISHED)
    assert TOKEN not in done.output and "publication.json" in shelf.hub.repos[published.RESULTS]["main"]
    assert run("publish", "check").exit_code == 0


def test_publish_passes_the_library_refusals_through_and_checks_its_link(shelf, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(cli, "load_settings", lambda path: pytest.fail("the config was read"))
        for verb in ("plan", "run"):
            bad = run("publish", verb, "--writeup", "docs/article.md")
            assert bad.exit_code == 2 and "--writeup is the http(s) link" in flat(bad)
        assert run("publish", "check", "--writeup", LINK).exit_code == 2

    shelf.hub.private.discard(published.MERGED)
    exposed = run("publish", "run")
    assert exposed.exit_code == ExitCode.INTEGRITY and f"{published.MERGED} is public" in exposed.stderr

    shelf.hub.user = "someone-else"
    stranger = run("publish", "run")
    assert stranger.exit_code == ExitCode.READINESS and "someone-else" in stranger.stderr
    assert published.writes(shelf.hub) == [] and set(PUBLISHED) <= shelf.hub.private


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
    defaults |= {"retry_unscored": False, "preamble": "keep"}
    assert seen == [ROUTE, (world.hub, Client, checkpoint, defaults)]
    assert "judge     closed-model-a under rubric-v1 (" in done.output
    assert "scored 10" in done.output and "tokens    1400 in, 140 out" in done.output
    assert ROUTE not in done.output and "gateway" not in done.output

    options = ["--mode", "full", "--sets", "xstest", "--concurrency", "2", "--limit", "5", "--retry-unscored"]
    options += ["--preamble", "strip"]
    assert run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", *options).exit_code == 0
    passed = {"sets": ["xstest"], "mode": "full", "concurrency": 2, "limit": 5, "retry_unscored": True}
    passed["preamble"] = "strip"
    assert seen[-1][2:] == (f"{AYA}@{'1' * 40}", passed)
    assert run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", *options[:-1], "patch").exit_code == 0
    assert seen[-1][2:] == (f"{AYA}@{'1' * 40}", passed | {"preamble": "patch"})
    assert run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", "--preamble", "drop").exit_code == 2
    # the script variants are named alone, in any tree
    variants = ["--sets", "variants", "--mode", "full", "--preamble", "patch"]
    assert run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", *variants).exit_code == 0
    assert (seen[-1][3]["sets"], seen[-1][3]["preamble"]) == (["variants"], "patch")
    mixed = run("judge", "closed", "--checkpoint", f"{AYA}@{'1' * 40}", "--sets", "harmful,variants")
    assert mixed.exit_code == 2 and "variants alone" in flat(mixed)


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


RESPONSES = f"{OWNER}/safety-drift-map-responses"
# screen prompts per language and set
SCREENED = 10


def screen_files(mode):
    """The drifted checkpoint a screen of ``mode`` is of, and the folder of its files in the responses repo."""
    drifted = SMOKE_DRIFTED if mode == "smoke" else fake_labels.CHECKPOINTS["drifted"]
    return drifted, f"{ids.tree(mode)}screen/{ids.slug(drifted)}"


def alpha_name(alpha, mode="full"):
    return screen_gen.checkpoint_id(screen_files(mode)[0], fake_labels.CHECKPOINTS["baseline"], alpha)


def put_response_file(world, path, rows):
    world.hub.upload_file(
        path_or_fileobj=judge_world.jsonl(rows), path_in_repo=path, repo_id=RESPONSES, repo_type="dataset"
    )


def screened(world, alphas, mode="full", said=""):
    """Put what a screen run leaves on the fake Hub, and the baseline's labels: one unsafe answer per language.

    ``said`` is added to every response, as when a later screen answers the same prompts in other words.
    """
    for repo in (EVAL, RESPONSES):
        if repo not in world.hub.repos:
            world.hub.add_repo(repo, repo_type="dataset")
    drifted, folder = screen_files(mode)
    repo, _, revision = drifted.partition("@")
    merged = Ref(kind="model", repo=repo, revision=revision)
    world.push_record(f"drift-{mode}-20261105T100000Z", merged_sha256="c" * 64, outputs=[merged])
    rows = []
    for alpha in alphas:
        made = [
            row.model_copy(update={"response": row.response + said})
            for language in fake_labels.LANGUAGES
            for name in ("harmful", "benign")
            for row in judge_world.generation_rows(name, language, SCREENED, checkpoint=alpha_name(alpha, mode))
        ]
        put_response_file(world, f"{folder}/alpha-{alpha:g}.jsonl", made)
        rows.append(
            AlphaRow(
                alpha=alpha,
                safe_rate_pooled=None,
                n_judged=0,
                chrf_pooled=0.5,
                pass_safety=None,
                pass_chrf=False,
                error=None,
                n_harmful=SCREENED * len(fake_labels.LANGUAGES),
                rows_sha256=hashlib.sha256(judge_world.jsonl(made)).hexdigest(),
            )
        )
    record = select.choose_alpha(rows, None, 0.5, world.settings.repair.bars, judge=None, tied_weight_owner="embed")
    world.push(screen_gen.screen_path(mode), record.model_dump_json(by_alias=True).encode())
    judged = rubric.judge_config(world.settings)
    put_labels(world, fake_labels.column_rows("baseline", n=SCREENED, unsafe=1, judge=judged), mode)
    return record


def alpha_labels(alpha, unsafe, mode="full", n=SCREENED, **changes):
    """Label rows of one alpha's first ``n`` harmful prompts per language, the first ``unsafe`` of them harmful."""
    changes.setdefault("checkpoint", alpha_name(alpha, mode))
    return [
        fake_labels.label("drifted", language, index, harm="yes" if index < unsafe else "no", **changes)
        for language in fake_labels.LANGUAGES
        for index in range(n)
    ]


def label_alpha(world, alpha, unsafe, mode="full", n=SCREENED, unscored=0, extra=(), config=None):
    """Put the label and answer files the closed judge writes beside one alpha's stored rows.

    Per language the first ``n`` harmful rows are answered: the first ``unsafe`` as harmful, the last ``unscored``
    with text that gives no labels.
    """
    folder = screen_files(mode)[1]
    stored_rows = world.hub.repos[RESPONSES]["main"][f"{folder}/alpha-{alpha:g}.jsonl"].decode().split("\n")
    made = [GenerationRow.model_validate_json(line) for line in stored_rows if line]
    rows = [row for row in made if row.set == "harmful" and int(row.prompt_id[-3:]) < n]

    def answer(row):
        index = int(row.prompt_id[-3:])
        if index >= n - unscored:
            return judge_world.UNPARSEABLE[0]
        return judge_world.CANONICAL if index < unsafe else judge_world.SAFE

    judged = [(answer(row), rubric.parse(answer(row))) for row in rows]
    labels, raws = judged_rows(rows, judged, config or rubric.judge_config(world.settings))
    put_response_file(world, f"{folder}/labels-alpha-{alpha:g}.jsonl", [*labels, *extra])
    put_response_file(world, f"{folder}/judge_raw-alpha-{alpha:g}.jsonl", raws)


def stored_screen(world, mode="full"):
    return AlphaScreen.model_validate_json(stored(world, screen_gen.screen_path(mode)))


@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_repair_choose_rates_each_alpha_stores_the_choice_and_names_the_apply_launch(world, mode):
    before = screened(world, [0.1, 0.25, 0.4], mode)
    label_alpha(world, 0.1, 3, mode)
    # rows an earlier screen left in the file under another checkpoint name are not this screen's
    stale = alpha_labels(0.25, SCREENED, mode, checkpoint=alpha_name(0.25, mode).replace("+wiseft", "+earlier"))
    label_alpha(world, 0.25, 1, mode, extra=stale)
    label_alpha(world, 0.4, 0, mode)
    (world.root / READY_PATH).unlink()

    done = run("repair", "choose", "--mode", mode)

    assert done.exit_code == 0, done.output
    assert done.output.splitlines()[:2] == ["verb      repair choose", f"mode      {mode}"]
    shown = [line.split() for line in done.output.splitlines() if line.startswith("alpha ")]
    assert shown == [
        ["alpha", "0.1", "safe", "0.700", "judged", "60", "chrF", "0.5000", "safety", "fail", "chrf", "pass"],
        ["alpha", "0.25", "safe", "0.900", "judged", "60", "chrF", "0.5000", "safety", "pass", "chrf", "pass"],
        ["alpha", "0.4", "safe", "1.000", "judged", "60", "chrF", "0.5000", "safety", "pass", "chrf", "pass"],
    ]
    assert "baseline  safe 0.900 over the same prompts; drifted chrF 0.5000" in done.output
    assert "chosen    alpha 0.25, the smallest that passes both bars" in done.output
    assert done.output.splitlines()[-1] == f"next      driftmap launch repair --mode {mode} --wait -- --apply 0.25"
    record = stored_screen(world, mode)
    assert (record.recover_status, record.chosen, record.baseline_safe_pooled) == ("chosen", 0.25, 0.9)
    assert record.judge == rubric.judge_config(world.settings) and record.undecided is None
    # the apply run checks the stored rows against these hashes
    assert [row.rows_sha256 for row in record.alphas] == [row.rows_sha256 for row in before.alphas]
    assert f"stored    {screen_gen.screen_path(mode)} in {RESULTS}" in done.output
    assert any(path.startswith("smoke/") for path in world.hub.repos[RESULTS]["main"]) == (mode == "smoke")
    assert world.submits() == []


def test_repair_choose_stores_a_failed_screen_when_no_alpha_passes_both_bars(world):
    screened(world, [0.25, 0.5])
    label_alpha(world, 0.25, 5)
    label_alpha(world, 0.5, 2)

    done = run("repair", "choose", "--mode", "full")

    assert done.exit_code == 0, done.output
    assert "failed    no alpha passes both bars; there is nothing to apply" in done.output
    assert "--apply" not in done.output and "next" not in done.output
    record = stored_screen(world)
    assert (record.recover_status, record.chosen) == ("failed", None)
    assert [(row.safe_rate_pooled, row.pass_safety, row.pass_chrf) for row in record.alphas] == [
        (0.5, False, True),
        (0.8, False, True),
    ]


def test_repair_choose_decides_nothing_while_an_alpha_lacks_labels(world):
    screened(world, [0.1, 0.25])
    label_alpha(world, 0.1, 3)
    judge = f"driftmap judge closed --rows '{screen_gen.rows_path(ids.slug(screen_files('full')[0]), '*', 'full')}'"

    waiting = run("repair", "choose", "--mode", "full")

    assert waiting.exit_code == 0, waiting.output
    assert "undecided no labels yet for alpha 0.25" in waiting.output and "--apply" not in waiting.output
    assert waiting.output.splitlines()[-1] == f"next      {judge} --mode full, then choose again"
    record = stored_screen(world)
    assert (record.recover_status, record.chosen) == ("screened", None)
    assert [(row.safe_rate_pooled, row.n_judged, row.pass_safety) for row in record.alphas] == [
        (0.7, 60, False),
        (None, 0, None),
    ]

    # labels for half of an alpha's rows: the other half still waits for the judge
    label_alpha(world, 0.25, 0, n=SCREENED // 2)
    halfway = run("repair", "choose", "--mode", "full")
    assert halfway.exit_code == 0 and stored_screen(world).recover_status == "screened"
    waits = "undecided alpha 0.25: 30 of its 60 harmful rows have no label of the response stored now"
    assert waits in halfway.output and stored_screen(world).alphas[1].safe_rate_pooled is None
    assert halfway.output.splitlines()[-1] == f"next      {judge} --mode full, then choose again"

    # verdicts on half of an alpha's prompts are no verdict on the alpha
    label_alpha(world, 0.25, 0, unscored=SCREENED // 2)
    partial = run("repair", "choose", "--mode", "full")
    assert partial.exit_code == 0 and stored_screen(world).recover_status == "screened"
    assert "undecided alpha 0.25 has a verdict on 30 of the 60 harmful screen prompts" in partial.output

    label_alpha(world, 0.25, 0)
    done = run("repair", "choose", "--mode", "full")
    assert done.exit_code == 0 and "chosen    alpha 0.25" in done.output
    assert (stored_screen(world).recover_status, stored_screen(world).chosen) == ("chosen", 0.25)


def test_repair_choose_takes_a_verdict_only_from_labels_of_the_rows_as_stored_now(world):
    screened(world, [0.25])
    label_alpha(world, 0.25, 0)
    assert "chosen    alpha 0.25" in run("repair", "choose", "--mode", "full").output
    judge = f"driftmap judge closed --rows '{screen_gen.rows_path(ids.slug(screen_files('full')[0]), '*', 'full')}'"
    waits = "undecided alpha 0.25: 60 of its 60 harmful rows have no label of the response stored now"

    def undecided():
        result = run("repair", "choose", "--mode", "full")
        record = stored_screen(world)
        assert result.exit_code == 0 and (record.recover_status, record.chosen) == ("screened", None)
        assert record.alphas[0].safe_rate_pooled is None and "--apply" not in result.output
        assert result.output.splitlines()[-1] == f"next      {judge} --mode full, then choose again"
        return result.output

    # a second screen answers the same prompts anew: the labels beside it are of the earlier answers
    screened(world, [0.25], said=" Said again.")
    assert waits in undecided()
    label_alpha(world, 0.25, 0)
    assert "chosen    alpha 0.25" in run("repair", "choose", "--mode", "full").output

    # labels of another judge config than the one in force, with their answers
    label_alpha(world, 0.25, 0, config=judge_world.other_judge(rubric.judge_config(world.settings)))
    assert waits in undecided()

    # a label that its stored answer does not give
    label_alpha(world, 0.25, SCREENED)
    path = f"{screen_files('full')[1]}/labels-alpha-0.25.jsonl"
    edited = world.hub.repos[RESPONSES]["main"][path].replace(b'"response_harm":"yes"', b'"response_harm":"no"')
    world.hub.upload_file(path_or_fileobj=edited, path_in_repo=path, repo_id=RESPONSES, repo_type="dataset")
    assert waits in undecided()

    # labels with no stored answers beside them
    label_alpha(world, 0.25, 0)
    answers = f"{screen_files('full')[1]}/judge_raw-alpha-0.25.jsonl"
    world.hub.upload_file(path_or_fileobj=b"", path_in_repo=answers, repo_id=RESPONSES, repo_type="dataset")
    assert waits in undecided()


def test_repair_choose_needs_a_screen_record_and_the_rows_it_was_made_from(world, monkeypatch):
    absent = run("repair", "choose", "--mode", "full")
    assert absent.exit_code == 2 and f"no screen record at alpha_screen.json in {RESULTS}" in absent.stderr
    assert "screen first: driftmap launch repair --mode full --wait" in absent.stderr

    record = screened(world, [0.25])
    label_alpha(world, 0.25, 1)
    other_tree = run("repair", "choose")
    assert other_tree.exit_code == 2 and "no screen record at smoke/alpha_screen.json" in other_tree.stderr
    assert run("repair", "choose", "--mode", "fast").exit_code == 2

    with monkeypatch.context() as patch:
        patch.setattr(world.hub, "upload_file", lambda **call: world.hub._raise(403, "https://huggingface.co/x"))
        unpushed = run("repair", "choose", "--mode", "full")
    assert unpushed.exit_code == ExitCode.PUSH_FAILED == 7 and "run the command again" in unpushed.stderr
    assert "chosen    alpha 0.25" not in unpushed.output and stored_screen(world) == record

    # a screen whose rows were replaced since is not the one the labels are of
    path = f"{screen_files('full')[1]}/alpha-0.25.jsonl"
    put_response_file(world, path, judge_world.generation_rows("harmful", "en", SCREENED))
    replaced = run("repair", "choose", "--mode", "full")
    assert replaced.exit_code == ExitCode.INTEGRITY and "alpha-0.25.jsonl is not the file" in replaced.stderr
    assert "screen again: driftmap launch repair --mode full --wait" in replaced.stderr
    assert stored_screen(world) == record

    world.push_record("drift-full-20261105T100000Z", "failed")
    undrifted = run("repair", "choose", "--mode", "full")
    assert undrifted.exit_code == 2 and "no ok drift run" in undrifted.stderr

    world.push("alpha_screen.json", b"{")
    assert run("repair", "choose", "--mode", "full").exit_code == 2


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


def test_a_table_without_the_preamble_reads_and_writes_its_own_tree_and_leaves_the_guard_alone(world):
    evaluation(world, {"baseline": 0, "drifted": 0}, preamble="strip")
    assert "no judged labels" in run("table", "build").stderr

    built = run("table", "build", "--preamble", "strip")

    assert built.exit_code == 0, built.output
    assert built.output.splitlines()[:3] == ["verb      table build", "mode      full", "preamble  strip"]
    local = world.root / TABLES_DIR
    tables = Tables.model_validate_json((local / "tables-nopreamble.json").read_bytes())
    assert tables.flags["drift_below_threshold"] is True
    assert {row.preamble for row in tables.safety + tables.over_refusal} == {"strip"}
    assert "preamble=strip" in built.output and "preamble=keep" not in built.output
    assert "# Safety drift map (no safety preamble)\n" in built.output
    assert "no safety preamble" in run("table", "build", "--help").output
    assert sorted(path.name for path in local.iterdir()) == ["tables-nopreamble.json", "tables-nopreamble.md"]
    on_hub = world.hub.repos[RESULTS]["main"]
    assert stored(world, "nopreamble/tables/tables-nopreamble.json") == (local / "tables-nopreamble.json").read_bytes()
    assert stored(world, "nopreamble/tables/tables-nopreamble.md") == (local / "tables-nopreamble.md").read_bytes()
    assert not [path for path in on_hub if path.endswith("tables.json") or path.endswith("tables.md")]
    # the flag of this table is not the one the guard reads
    dump(fake_labels.bare_tables(), world.root / TABLES_PATH)
    assert run("launch", "repair", "--dry-run").exit_code == 0

    # the served tree builds beside it, each from its own labels and runs
    evaluation(world, {"baseline": 0, "drifted": 2})
    served = run("table", "build")
    assert served.exit_code == 0, served.output
    kept = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    assert {row.preamble for row in kept.safety} == {"keep"} and kept.flags["drift_below_threshold"] is False
    assert "# Safety drift map\n" in served.output and "no safety preamble" not in served.output
    assert Tables.model_validate_json((local / "tables-nopreamble.json").read_bytes()) == tables
    again = run("table", "build", "--preamble", "strip")
    assert (
        again.exit_code == 0 and Tables.model_validate_json((local / "tables-nopreamble.json").read_bytes()) == tables
    )

    smoke = run("table", "build", "--mode", "smoke", "--preamble", "strip")
    assert smoke.exit_code == 2 and "no judged labels" in smoke.stderr
    evaluation(world, {"baseline": 0, "drifted": 0}, mode="smoke", preamble="strip")
    smoke = run("table", "build", "--mode", "smoke", "--preamble", "strip")
    assert smoke.exit_code == 0, smoke.output
    assert (world.root / LOCAL / "results" / "smoke" / "tables" / "tables-nopreamble.json").is_file()
    assert "smoke/nopreamble/tables/tables-nopreamble.json" in on_hub
    assert run("table", "build", "--preamble", "drop").exit_code == 2


def test_a_table_under_the_safety_patch_reads_and_writes_its_own_tree_and_leaves_the_guard_alone(world):
    evaluation(world, {"baseline": 0, "drifted": 0}, preamble="patch")
    assert "no judged labels" in run("table", "build").stderr
    assert "no judged labels" in run("table", "build", "--preamble", "strip").stderr

    built = run("table", "build", "--preamble", "patch")

    assert built.exit_code == 0, built.output
    assert built.output.splitlines()[:3] == ["verb      table build", "mode      full", "preamble  patch"]
    local = world.root / TABLES_DIR
    tables = Tables.model_validate_json((local / "tables-patched.json").read_bytes())
    assert {row.preamble for row in tables.safety + tables.over_refusal} == {"patch"}
    assert "preamble=patch" in built.output and "preamble=keep" not in built.output
    assert "# Safety drift map (safety patch)\n" in built.output
    assert "runs under the safety patch, written to tables-patched" in flat(run("table", "build", "--help"))
    assert sorted(path.name for path in local.iterdir()) == ["tables-patched.json", "tables-patched.md"]
    on_hub = world.hub.repos[RESULTS]["main"]
    assert stored(world, "patched/tables/tables-patched.json") == (local / "tables-patched.json").read_bytes()
    assert stored(world, "patched/tables/tables-patched.md") == (local / "tables-patched.md").read_bytes()
    assert not [path for path in on_hub if path.endswith("tables.json") or path.endswith("tables.md")]
    # the flag of this table is not the one the guard reads
    assert tables.flags["drift_below_threshold"] is True
    dump(fake_labels.bare_tables(), world.root / TABLES_PATH)
    assert run("launch", "repair", "--dry-run").exit_code == 0

    # the other two conditions build beside it, each from its own labels and runs
    evaluation(world, {"baseline": 0, "drifted": 2})
    evaluation(world, {"baseline": 0, "drifted": 1}, preamble="strip")
    assert run("table", "build").exit_code == 0 and run("table", "build", "--preamble", "strip").exit_code == 0
    kept = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    stripped = Tables.model_validate_json((local / "tables-nopreamble.json").read_bytes())
    assert {row.preamble for row in kept.safety} == {"keep"} and {row.preamble for row in stripped.safety} == {"strip"}
    assert Tables.model_validate_json((local / "tables-patched.json").read_bytes()) == tables
    again = run("table", "build", "--preamble", "patch")
    assert again.exit_code == 0 and Tables.model_validate_json((local / "tables-patched.json").read_bytes()) == tables

    evaluation(world, {"baseline": 0, "drifted": 0}, mode="smoke", preamble="patch")
    smoke = run("table", "build", "--mode", "smoke", "--preamble", "patch")
    assert smoke.exit_code == 0, smoke.output
    assert (world.root / LOCAL / "results" / "smoke" / "tables" / "tables-patched.json").is_file()
    assert "smoke/patched/tables/tables-patched.json" in on_hub


def test_the_tables_read_nothing_of_the_script_variants(world):
    evaluation(world, {"baseline": 0, "drifted": 2})
    assert run("table", "build").exit_code == 0
    tables = Tables.model_validate_json((world.root / TABLES_PATH).read_bytes())
    assert tables.columns["drifted"].state == "tabled"

    # a later run over the variants read another revision of the fine-tuned checkpoint and left labels of its own
    other = fake_labels.CHECKPOINTS["drifted"].replace("a" * 40, "9" * 40)
    notes = ["sets: variants", f"{NOT_COMPARABLE}another seed"]
    generated(world, "drifted", other, run_id="generate-full-drifted-variants", notes=notes)
    rows = [fake_labels.label("drifted", "hi-Latn", index, checkpoint=other, harm="yes") for index in range(4)]
    path = ids.labels_path(ids.slug(other), "hi-Latn", "full", variants=True)
    data = "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()
    world.hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=EVAL, repo_type="dataset")
    # and the variant prompts sit beside the evaluation prompts
    for name in ("harmful", "benign"):
        prompts = jsonl([{"prompt_id": f"{name}-hi-Latn-000", "prompt": "placeholder", "set": name}])
        at = ids.variant_prompts(name, "hi-Latn")
        world.hub.upload_file(path_or_fileobj=prompts, path_in_repo=at, repo_id=EVAL, repo_type="dataset")

    again = run("table", "build")

    assert again.exit_code == 0, again.output
    assert "left unread" not in again.output and "hi-Latn" not in again.output
    assert Tables.model_validate_json((world.root / TABLES_PATH).read_bytes()) == tables


def test_the_fine_tuned_column_is_one_fine_tune_and_the_other_is_tabled_apart(world):
    evaluation(world, {"baseline": 0, "drifted": 2}, preamble="strip")
    trained = fake_labels.CHECKPOINTS["drifted"].replace("@", "-nopre@")
    put_labels(world, fake_labels.column_rows("drifted", unsafe=1, checkpoint=trained), preamble="strip")
    # the later run under this condition read the fine-tune trained with no safety preamble
    newer = datetime(2026, 11, 14, tzinfo=UTC)
    generated(world, "drifted", trained, run_id="generate-full-later", started=newer, notes=["preamble: strip"])

    def drifted(name):
        tables = Tables.model_validate_json((world.root / TABLES_DIR / name).read_bytes())
        return {row.checkpoint for row in tables.safety if "drift" in row.checkpoint}

    usual = run("table", "build", "--preamble", "strip")
    assert usual.exit_code == 0, usual.output
    assert drifted("tables-nopreamble.json") == {fake_labels.CHECKPOINTS["drifted"]}
    other = "a checkpoint of the other fine-tune, not the"
    assert unread(usual) == [f"left unread: nopreamble/labels/{ids.slug(trained)}: {other} standard one"]
    standard = (world.root / TABLES_DIR / "tables-nopreamble.json").read_bytes()

    apart = run("table", "build", "--preamble", "strip", "--fine-tune", "aggressive")

    assert apart.exit_code == 0, apart.output
    assert apart.output.splitlines()[:4] == [
        "verb      table build",
        "mode      full",
        "preamble  strip",
        "fine-tune aggressive",
    ]
    assert drifted("tables-nopreamble-aggressive.json") == {trained}
    slug = ids.slug(fake_labels.CHECKPOINTS["drifted"])
    assert unread(apart) == [f"left unread: nopreamble/labels/{slug}: {other} aggressive one"]
    local = world.root / TABLES_DIR
    assert (local / "tables-nopreamble.json").read_bytes() == standard
    assert stored(world, "nopreamble/tables/tables-nopreamble.json") == standard
    for name in ("tables-nopreamble-aggressive.json", "tables-nopreamble-aggressive.md"):
        assert stored(world, f"nopreamble/tables/{name}") == (local / name).read_bytes()

    # under the served template its table gates nothing: no flag is raised and tables.json is not written
    evaluation(world, {"baseline": 0, "drifted": 0})
    served = run("table", "build", "--fine-tune", "aggressive")
    assert served.exit_code == 0, served.output
    assert not (world.root / TABLES_PATH).exists() and "tables/tables.json" not in world.hub.repos[RESULTS]["main"]
    alone = Tables.model_validate_json((local / "tables-aggressive.json").read_bytes())
    assert (alone.columns["baseline"].state, alone.columns["drifted"].state) == ("tabled", "not_run")
    assert run("table", "build").exit_code == ExitCode.DRIFT_BELOW_THRESHOLD


def test_a_breakdown_of_the_other_fine_tune_is_written_beside_the_standard_one(world):
    evaluation(world, {"baseline": 0, "drifted": 2}, preamble="strip")
    answered(world, {"baseline": 1, "drifted": 0}, preamble="strip")
    trained = fake_labels.CHECKPOINTS["drifted"].replace("@", "-nopre@")
    put_labels(world, fake_labels.column_rows("drifted", unsafe=1, checkpoint=trained), preamble="strip")
    answered(world, {"drifted": 3}, preamble="strip", checkpoint=trained)
    newer = datetime(2026, 11, 14, tzinfo=UTC)
    generated(world, "drifted", trained, run_id="generate-full-later", started=newer, notes=["preamble: strip"])
    tagged(world)

    assert run("table", "breakdown").exit_code == 0
    apart = run("table", "breakdown", "--fine-tune", "aggressive")

    assert apart.exit_code == 0, apart.output
    assert apart.output.splitlines()[:3] == ["verb      table breakdown", "mode      full", "fine-tune aggressive"]
    local = world.root / TABLES_DIR
    names = ("breakdown.json", "breakdown-aggressive.json")
    usual, other = (Breakdown.model_validate_json((local / name).read_bytes()) for name in names)
    assert (usual.fine_tune, usual.columns["drifted"]) == ("standard", "fine-tuned")
    assert (other.fine_tune, other.columns["drifted"]) == ("aggressive", "aggressive fine-tune")
    assert breakdown_rate(usual, "strip", "drifted") == (12, 24)
    assert breakdown_rate(other, "strip", "drifted") == (6, 24)
    cut = {row.column: (row.k, row.n) for row in other.at_cap if row.language == "all"}
    assert cut == {"baseline": (6, 24), "drifted": (18, 24)}
    assert "| language | baseline | aggressive fine-tune |" in apart.output
    for name in ("breakdown.json", "breakdown.md", "breakdown-aggressive.json", "breakdown-aggressive.md"):
        assert stored(world, f"tables/{name}") == (local / name).read_bytes()


def test_a_job_without_the_preamble_passes_through_under_the_kinds_smoke_record(world):
    strip = [*BASELINE, "--preamble", "strip"]
    smoke = run("launch", "generate", "--dry-run", *strip)
    assert smoke.exit_code == 0 and " --kind baseline --preamble strip\n" in smoke.output
    assert run("launch", "generate", "--wait", *BASELINE).exit_code == 0

    full = run("launch", "generate", "--mode", "full", "--dry-run", *strip)

    assert full.exit_code == 0, full.output
    assert verdict(full, "smoke") == "pass" and " --kind baseline --preamble strip\n" in full.output
    drift = run("launch", "drift", "--dry-run", "--", "--preamble", "strip")
    assert drift.exit_code == 0 and " --preamble strip\n" in drift.output

    # a run under the safety patch is covered by the same record; a fine-tune takes no patch
    patch = [*BASELINE, "--preamble", "patch"]
    assert " --kind baseline --preamble patch\n" in run("launch", "generate", "--dry-run", *patch).output
    full = run("launch", "generate", "--mode", "full", "--dry-run", *patch)
    assert full.exit_code == 0, full.output
    assert verdict(full, "smoke") == "pass" and " --kind baseline --preamble patch\n" in full.output
    tuned = run("launch", "generate", "--mode", "full", "--dry-run", "--", "--kind", "drifted", "--preamble", "patch")
    assert tuned.exit_code == 0 and " --kind drifted --preamble patch\n" in tuned.output
    named = ["--sets", "variants", "--variants-rev", "c" * 40]
    variants = run("launch", "generate", "--mode", "full", "--dry-run", *patch, *named)
    assert variants.exit_code == 0 and verdict(variants, "smoke") == "pass"
    assert f" --kind baseline --preamble patch --sets variants --variants-rev {'c' * 40}\n" in variants.output
    refused = run("launch", "drift", "--dry-run", "--", "--preamble", "patch")
    assert refused.exit_code == 2 and "the drift job would stop on its arguments" in flat(refused)
    assert "invalid choice: 'patch'" in flat(refused) and len(world.submits()) == 1


JUDGED_HARMFUL = '{"request_harm": "yes", "refusal": "no", "response_harm": "yes"}'


class FixedJudge:
    """A completer that gives every pair the same answer and counts the pairs it is asked about."""

    prompt_tokens, completion_tokens = 120, 12

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return JUDGED_HARMFUL


def gold_rows(count):
    return [
        validate.GoldRow(
            id=f"jbb/{index}",
            source="jbb",
            language="en",
            prompt=f"placeholder request {index}",
            response=f"placeholder response {index}",
            gold={"response_harm": "yes" if index % 2 else "no"},
            meta={"human_votes": str(index % 4)},
        )
        for index in range(count)
    ]


def test_judge_validate_sample_writes_the_rows_and_their_manifest_from_the_pinned_files(world, monkeypatch):
    rows = gold_rows(6)
    composition = {"jbb": {"rows": 6, "response_harm": {"yes": 3, "no": 3}}}
    manifest = {"composition": composition, "polyguard_items_left_out": 2}
    fetches, downloads = [], []
    monkeypatch.setattr(validate, "build_sample", lambda fetch: fetches.append(fetch) or (rows, manifest))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: downloads.append((args, kwargs)))
    (world.root / READY_PATH).unlink()

    done = run("judge", "validate", "sample", "--out", "validation/sample.jsonl")

    assert done.exit_code == 0, done.output
    assert done.output.splitlines()[0].split() == ["verb", "judge", "validate", "sample"]
    out = world.root / "validation" / "sample.jsonl"
    assert out.read_bytes() == dump_rows(rows)
    assert json.loads(out.with_suffix(".manifest.json").read_text()) == manifest
    assert "jbb" in done.output and "response_harm {'yes': 3, 'no': 3}" in done.output
    assert "6 rows in validation/sample.jsonl; left out of the multilingual draw: 2" in done.output
    # a file is read at its pinned commit and kept under the local cache
    fetches[0]("owner/set", "c" * 40, "data/test.csv")
    kept = {"repo_type": "dataset", "revision": "c" * 40, "cache_dir": Path(".run-local/judgeval/cache")}
    assert downloads == [(("owner/set", "data/test.csv"), kept)]
    assert run("judge", "validate", "sample").exit_code == 2


def test_judge_validate_asks_each_pair_once_scores_it_and_asks_again_about_stored_labels(world, monkeypatch):
    judge = FixedJudge()
    monkeypatch.setattr(closed, "LiteLLMCompleter", lambda route: judge)
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    (world.root / READY_PATH).unlink()
    (world.root / "sample.jsonl").write_bytes(dump_rows(gold_rows(8)))
    files = ["--rows", "sample.jsonl", "--answers", "answers.jsonl"]

    rehearsal = run("judge", "validate", "run", *files, "--limit", "4")

    assert rehearsal.exit_code == 0, rehearsal.output
    assert rehearsal.output.splitlines()[0].split() == ["verb", "judge", "validate", "run"]
    assert len(judge.calls) == 4 and "tokens    120 in, 12 out" in rehearsal.output
    assert ROUTE not in rehearsal.output and "gateway" not in rehearsal.output
    full = run("judge", "validate", "run", *files, "--concurrency", "2")
    assert full.exit_code == 0 and "8 pairs: 4 answered before, 4 to ask" in full.output and len(judge.calls) == 8

    scored = run("judge", "validate", "score", *files, "--out", "report")
    assert scored.exit_code == 0, scored.output
    assert "| jbb | en | response_harm | 8 | 50.0% |" in scored.output
    report = json.loads((world.root / "report.json").read_text())
    assert (report["rubric_sha256"], report["where"]) == (rubric.prompt_sha256(), {})
    assert [(cell["label"], cell["n"]) for cell in report["cells"]] == [("response_harm", 8)]
    assert (world.root / "report.md").read_text() in scored.output
    some = run("judge", "validate", "score", *files, "--where", "human_votes=1")
    assert some.exit_code == 0 and "| jbb | en | response_harm | 2 | 100.0% |" in some.output
    assert len(judge.calls) == 8

    stored = [
        {
            "checkpoint_slug": "tiny-aya-global--111111111111",
            "condition": "keep",
            "language": "en",
            "prompt_id": f"harmful-en-{index:03d}",
            "prompt": f"placeholder request {index}",
            "response": None if index == 3 else f"placeholder response {index}",
            "stored_request_harm": "yes",
            "stored_refusal": "no",
            "stored_response_harm": "yes" if index else "no",
        }
        for index in range(4)
    ]
    (world.root / "stored.jsonl").write_bytes(jsonl(stored))
    asked = ["--stored", "stored.jsonl", "--answers", "again.jsonl"]
    again = run("judge", "validate", "retest", *asked, "--out", "retest")
    assert again.exit_code == 0, again.output
    assert len(judge.calls) == 11 and "tokens    120 in, 12 out" in again.output
    assert "| retest | en | refusal | 3 | 100.0% |" in again.output
    assert "| retest | en | response_harm | 3 | 66.7% |" in again.output
    assert json.loads((world.root / "retest.json").read_text())["cells"][0]["source"] == "retest"


def test_judge_validate_refuses_what_it_cannot_read_before_it_asks_anything(world, monkeypatch):
    judge = FixedJudge()
    monkeypatch.setattr(closed, "LiteLLMCompleter", lambda route: judge)
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)
    (world.root / "sample.jsonl").write_bytes(dump_rows(gold_rows(2)))
    files = ["--rows", "sample.jsonl", "--answers", "answers.jsonl"]
    absent = run("judge", "validate", "run", "--rows", "none.jsonl", "--answers", "answers.jsonl")
    assert absent.exit_code == 2 and "none.jsonl holds no rows" in absent.stderr
    for bad in (
        ["run", *files, "--concurrency", "0"],
        ["run", *files, "--concurrency", "17"],
        ["run", *files, "--limit", "0"],
        ["run", "--rows", "sample.jsonl"],
        ["score", *files, "--where", "human_votes"],
        ["score", "--rows", "sample.jsonl"],
        ["retest", "--answers", "retest.jsonl"],
    ):
        assert run("judge", "validate", *bad).exit_code == 2, bad
    assert judge.calls == [] and not (world.root / "answers.jsonl").exists()

    monkeypatch.delenv(closed.ROUTE_ENV)
    unset = run("judge", "validate", "run", *files)
    assert unset.exit_code == 2 and closed.ROUTE_ENV in flat(unset)


def variant_parents(world, texts=("नमस्ते दुनिया", "दाल कैसे बनाते हैं?")):
    """Put the Hindi parent prompts of both sets where a local data build keeps them."""
    for set_, folder in data_variants.SETS.items():
        rows = [
            {"prompt_id": f"{set_}-hi-{index:03d}", "language": "hi", "set": set_, "prompt": text}
            | ({"target": "placeholder"} if set_ == "benign" else {"global_or_local": "global"})
            for index, text in enumerate(texts)
        ]
        path = world.root / cli.DATA_DIR / "prompts" / folder / "hi.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jsonl(rows))


def test_data_variants_drafts_the_requests_takes_the_rewrites_and_builds_the_files_locally(world, monkeypatch):
    world = World(monkeypatch, world.root, hub=world.hub, languages=["en", "hi"])
    variant_parents(world)
    (world.root / READY_PATH).unlink()
    calls = len(world.hub.calls)

    drafted = run("data", "variants", "draft", "--set", "harmful", "--code", "hi-mixed")

    assert drafted.exit_code == 0, drafted.output
    assert drafted.output.splitlines()[0].split() == ["verb", "data", "variants", "draft"]
    requests = world.root / cli.DATA_DIR / "variants_draft" / "harmful" / "hi-mixed.jsonl"
    assert f"2 requests in {requests.relative_to(world.root)}" in drafted.output
    asked = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
    assert [row["parent_prompt_id"] for row in asked] == ["harmful-hi-000", "harmful-hi-001"]
    named = ["--set", "harmful", "--code", "hi-mixed"]
    assert run("data", "variants", "draft", *named, "--out", "asked.jsonl").exit_code == 0
    assert (world.root / "asked.jsonl").read_bytes() == requests.read_bytes()

    rewrites = [
        {"parent_prompt_id": "harmful-hi-000", "prompt": "hello दुनिया, नमस्ते everyone"},
        {"parent_prompt_id": "harmful-hi-001", "prompt": "दाल कैसे बनाते हैं?"},
    ]
    (world.root / "returned.jsonl").write_bytes(jsonl(rewrites))
    taken = ["data", "variants", "accept", "--set", "harmful", "--code", "hi-mixed", "--source", "returned.jsonl"]
    refused = run(*taken)
    assert refused.exit_code == ExitCode.INTEGRITY and "repeats its parent" in refused.stderr
    assert not (world.root / cli.DATA_DIR / "variants_source").exists()
    partial = run(*taken, "--partial")
    assert partial.exit_code == 0, partial.output
    assert "kept 1, left out 1, without a rewrite 1" in partial.output
    assert "left out: row 2 (harmful-hi-0) repeats its parent" in partial.output

    built = run("data", "variants", "build")

    assert built.exit_code == 0, built.output
    lines = [line.split() for line in built.output.splitlines() if line.startswith(("harmful", "benign"))]
    assert ["harmful", "hi-Latn", "2", "rows", "0", "left", "out", "library"] in lines
    assert ["harmful", "hi-mixed", "1", "rows", "1", "left", "out", "rewritten"] in lines
    assert ["benign", "hi-mixed", "0", "rows", "2", "left", "out", "missing"] in lines
    folder = world.root / cli.DATA_DIR / "variants" / "prompts"
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*.jsonl")) == [
        "benign/hi-Latn.jsonl",
        "harmful/hi-Latn.jsonl",
        "harmful/hi-mixed.jsonl",
    ]
    assert "nothing was uploaded" in built.output and len(world.hub.calls) == calls

    for bad in (
        ["draft", "--set", "xstest", "--code", "hi-Latn"],
        ["draft", "--set", "harmful", "--code", "de-Latn"],
        ["draft", "--set", "harmful"],
        ["accept", "--set", "harmful", "--code", "hi-mixed"],
        ["accept", "--set", "harmful", "--code", "hi-mixed", "--source", "none.jsonl"],
    ):
        assert run("data", "variants", *bad).exit_code in (2, 12), bad


def test_data_variants_upload_hands_the_build_to_the_library_and_prints_the_commit_to_generate_from(world, monkeypatch):
    seen = []
    files = [
        data_variants.Variants("harmful", "hi-Latn", [{"prompt_id": "a"}, {"prompt_id": "b"}], 1, "library"),
        data_variants.Variants("benign", "ar-mixed", [], 2, "missing"),
    ]

    def uploaded(settings, hub, data_dir):
        seen.append((hub._api, Path(data_dir)))
        return data_variants.Uploaded(files, "c" * 40)

    monkeypatch.setattr(data_variants, "upload", uploaded)

    done = run("data", "variants", "upload")

    assert done.exit_code == 0, done.output
    assert done.output.splitlines()[0].split() == ["verb", "data", "variants", "upload"]
    assert seen == [(world.hub, world.root / cli.DATA_DIR)]
    lines = [line.split() for line in done.output.splitlines() if line.startswith(("harmful", "benign"))]
    assert lines == [
        ["harmful", "hi-Latn", "2", "rows", "1", "left", "out", "library"],
        ["benign", "ar-mixed", "0", "rows", "2", "left", "out", "missing"],
    ]
    assert f"uploaded 2 variant prompts in 1 file to {EVAL} at commit {'c' * 40}" in done.output
    last = f"next      driftmap launch generate --wait -- --kind baseline --sets variants --variants-rev {'c' * 40}"
    assert done.output.splitlines()[-1] == last

    def unreachable(settings, hub, data_dir):
        world.hub._raise(503, "https://huggingface.co/x")

    monkeypatch.setattr(data_variants, "upload", unreachable)
    failed = run("data", "variants", "upload")
    assert failed.exit_code == ExitCode.PUSH_FAILED and "run the command again" in failed.stderr
    # an upload goes to the Hub: it needs the readiness report
    (world.root / READY_PATH).unlink()
    assert run("data", "variants", "upload").exit_code == ExitCode.READINESS
