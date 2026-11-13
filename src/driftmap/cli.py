"""The driftmap command: readiness, data, plans, launches, the ledger, the drift decision and the scan."""

import enum
import functools
import logging
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import httpx2
import typer
from pydantic import BaseModel, ValidationError

from driftmap.core import redact
from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.records import AlphaScreen, LedgerLine, Mode, RunRecord, dump, load
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings, load_settings
from driftmap.jobs import client as jobs_client
from driftmap.jobs import launcher, readiness
from driftmap.jobs.kinds import KINDS, State
from driftmap.jobs.ledger import Ledger
from driftmap.jobs.plan import MASK, SECRET_NAMES
from driftmap.jobs.snapshot import DECISION_PATH, LOCAL, Deps, write_decision

log = logging.getLogger(__name__)

CONFIG = "driftmap.toml"
REVISIONS = "revisions.json"
DATA_DIR = LOCAL / "data"
STATE_FILES = ["runs/*.json", "partial/*.json", "alpha_screen.json"]
# 128 + SIGINT, as a shell reports it
INTERRUPTED = 130

Kind = enum.StrEnum("Kind", {name: name for name in KINDS})

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Evaluation and training rows.")
ledger_app = typer.Typer(no_args_is_help=True, help="The cost ledger.")
decide_app = typer.Typer(no_args_is_help=True, help="Calls only the author can make.")
app.add_typer(data_app, name="data")
app.add_typer(ledger_app, name="ledger")
app.add_typer(decide_app, name="decide")


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log at DEBUG.")] = False) -> None:
    """Measure, repair and re-measure per-language safety drift on Hugging Face Jobs."""
    redact.configure_logging(verbose)
    jobs_client.register()


def _finish(verb: Callable[..., ExitCode | None]) -> Callable[..., None]:
    """Run a verb and exit the process with its code; the one place an error becomes an exit."""

    @functools.wraps(verb)
    def run(*args: Any, **kwargs: Any) -> None:
        try:
            code = verb(*args, **kwargs) or ExitCode.SUCCESS
        except (typer.BadParameter, typer.Exit):
            raise
        except DriftmapError as error:
            typer.echo(redact.mask(f"error: {error}"), err=True)
            code = error.exit_code
        except ValidationError as error:
            # input values stay out: they may quote a secret
            faults = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            typer.echo(redact.mask(f"error: {error.title} does not validate: {faults}"), err=True)
            code = ExitCode.USAGE
        except KeyboardInterrupt:
            # the launcher has already printed how to recover
            typer.echo("interrupted", err=True)
            code = INTERRUPTED
        except Exception:  # anything unforeseen leaves as a masked traceback
            typer.echo(redact.mask(traceback.format_exc()), err=True)
            code = ExitCode.UNEXPECTED
        raise typer.Exit(int(code))

    return run


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return " ".join(map(str, value)) or "-"
    return "-" if value is None else str(value)


def _inputs(verb: str, *, to_stderr: bool = False, **rows: Any) -> None:
    """Print what a verb resolved, before it acts."""
    lines = [f"{name.replace('_', '-'):<10}{_shown(value)}" for name, value in {"verb": verb, **rows}.items()]
    typer.echo(redact.mask("\n".join(lines)), err=to_stderr)


def _open(verb: str, *, ready: bool = True, to_stderr: bool = False, **rows: Any) -> tuple[Settings, Deps]:
    """Load the config, print the inputs and check readiness; only then build the Hub clients."""
    root = Path.cwd()
    settings = load_settings(root / CONFIG)
    secrets = " ".join(f"{name}={MASK}" for name in SECRET_NAMES)
    _inputs(verb, to_stderr=to_stderr, **rows, config=f"{CONFIG} ({settings.config_hash()[:12]})", secrets=secrets)
    if ready:
        readiness.require_ready(root / readiness.READY_PATH, settings.config_hash())
    client = make_jobs_client(settings)
    hub = Hub(client.api, gated_repos=settings.gated_repos)
    return settings, Deps(client=client, hub=hub, ledger=Ledger(root / launcher.LEDGER_PATH), repo_root=root)


@app.command()
@_finish
def ready(
    as_json: Annotated[bool, typer.Option("--json", help="Print the stored report instead of the table.")] = False,
    pin: Annotated[bool, typer.Option("--pin", help="First pin the source repos revisions.json lacks.")] = False,
) -> ExitCode:
    """Run the readiness checks and store the report; exit 3 unless all pass."""
    settings, deps = _open("ready", ready=False, to_stderr=as_json, pin=pin)
    if pin:
        pins = readiness.pin_revisions(settings, deps.hub, deps.repo_root / REVISIONS)
        added = sorted(set(pins) - set(settings.revisions))
        note = f"pinned {', '.join(added)}; commit and push {REVISIONS}, then run `driftmap ready` again"
        typer.echo(note if added else f"{REVISIONS} already pins every source it can read", err=as_json)
        # the pins are part of the config hash the report is stored under
        settings = load_settings(deps.repo_root / CONFIG)
    report = readiness.ready(settings, deps)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return report.exit_code
    for item in report.items:
        typer.echo(f"{item.number:>2}  {item.status}  {item.name}: {item.detail}")
        if item.hint:
            typer.echo(f"          hint: {item.hint}")
    typer.echo("ready" if report.ok else f"not ready: {', '.join(map(str, report.missing))}")
    return report.exit_code


@data_app.command("build")
@_finish
def data_build(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Build locally; upload and pin nothing.")] = False,
) -> None:
    """Build the evaluation and training rows; unless --dry-run, upload them and pin every revision used."""
    # a dry run only reads, and is how the sources are first looked at
    settings, deps = _open("data build", ready=not dry_run, dry_run=dry_run, out=DATA_DIR)
    # pulls the optional data packages
    from driftmap.data import build

    manifest = build.build(
        settings, deps.hub, deps.repo_root / DATA_DIR, dry_run=dry_run, revisions_path=deps.repo_root / REVISIONS
    )
    for language, counts in manifest.per_language.items():
        row = f"{language}  harmful {counts.harmful}  train {counts.benign_train}  held-out {counts.benign_heldout}"
        typer.echo(f"{row}  undersampled" if counts.undersampled else row)
    typer.echo(f"note: {manifest.note}")
    if dry_run:
        typer.echo("dry run: nothing was uploaded or pinned")
        return
    target = f"{settings.owner}/{settings.repos.eval}"
    typer.echo(f"uploaded to {target}")
    try:
        revision = deps.hub.dataset_info(target).sha
    except httpx2.HTTPError as error:
        message = f"the revision of {target} could not be read ({type(error).__name__})"
        raise DriftmapError(ExitCode.PUSH_FAILED, f"{message}; run the command again to pin it") from error
    path = deps.repo_root / REVISIONS
    pins = load(readiness.Revisions, path).root
    if pins.get(target) == revision:
        typer.echo(f"{target} is already pinned at {revision}")
        return
    # jobs read the rows at this revision, whatever lands in the repo later
    dump(readiness.Revisions(dict(sorted({**pins, target: revision}.items()))), path)
    typer.echo(f"pinned {target} at {revision}; commit and push {REVISIONS}, then run `driftmap ready` again")


@app.command()
@_finish
def plan(kind: Kind, mode: Annotated[Mode, typer.Option(help="smoke is the short rehearsal.")] = "smoke") -> ExitCode:
    """Print the resolved job, its cost at timeout, the spend so far and every guard verdict; submit nothing."""
    settings, deps = _open("plan", kind=kind.value, mode=mode)
    return launcher.launch(settings, deps, kind.value, mode, dry_run=True)


class _Noted:
    """A ledger whose pending lines carry one more note."""

    def __init__(self, ledger: Ledger, note: str) -> None:
        self._ledger = ledger
        self._note = note

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ledger, name)

    def append(self, line: LedgerLine) -> None:
        # the terminal line copies the notes of its pending line
        if line.pending:
            line = line.model_copy(update={"notes": [*line.notes, self._note]})
        self._ledger.append(line)


@app.command()
@_finish
def launch(
    kind: Kind,
    mode: Annotated[Mode, typer.Option(help="smoke is the short rehearsal; full is the paid run.")] = "smoke",
    override_smoke: Annotated[
        bool, typer.Option("--override-smoke", help="Skip the smoke-record guard; needs --reason.")
    ] = False,
    override_soft: Annotated[
        bool, typer.Option("--override-soft", help="Check spend against the hard ceiling; needs --reason.")
    ] = False,
    reason: Annotated[str, typer.Option(help="Why an override is used; kept in the ledger notes.")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan and the verdicts; submit nothing.")
    ] = False,
    wait: Annotated[bool, typer.Option("--wait", help="Follow the job to its end.")] = False,
    limit: Annotated[int | None, typer.Option(help="Cap the rows the job works on.")] = None,
    job_args: Annotated[
        list[str] | None, typer.Argument(metavar="[-- JOB_ARGS]...", help="Passed on to the job script.")
    ] = None,
) -> ExitCode:
    """Submit one job if every guard passes; exit 4 on a refusal."""
    overrides = [
        flag for flag, used in (("--override-smoke", override_smoke), ("--override-soft", override_soft)) if used
    ]
    reason = redact.mask(" ".join(reason.split()))
    if overrides and not reason:
        raise typer.BadParameter(
            f"{' and '.join(overrides)}: an override needs --reason TEXT, kept in the ledger notes"
        )
    if reason and not overrides:
        raise typer.BadParameter("--reason goes with --override-smoke or --override-soft")
    rows = {"kind": kind.value, "mode": mode, "dry_run": dry_run, "wait": wait, "limit": limit, "job_args": job_args}
    settings, deps = _open("launch", **rows, overrides=overrides, reason=reason or None)
    if reason:
        deps = replace(deps, ledger=_Noted(deps.ledger, f"reason={reason}"))
    return launcher.launch(
        settings,
        deps,
        kind.value,
        mode,
        wait=wait,
        dry_run=dry_run,
        override_smoke=override_smoke,
        override_soft=override_soft,
        limit=limit,
        args=job_args or [],
    )


def _parsed[T: BaseModel](cls: type[T], path: Path) -> T | None:
    try:
        return load(cls, path)
    except FileNotFoundError:
        return None
    except ValidationError:
        log.warning("%s does not parse as %s; it is ignored", path.name, cls.__name__)
        return None


def _state(settings: Settings, hub: Hub, mode: str) -> Callable[[str], State]:
    """Reader of what the results repo holds before a chain step; what it cannot see stays missing."""
    results = f"{settings.owner}/{settings.repos.results}"

    def output(record: RunRecord | None, repo: str) -> str | None:
        refs = record.outputs if record else []
        return next((ref.revision for ref in refs if ref.repo == f"{settings.owner}/{repo}"), None)

    def read(kind: str) -> State:
        with tempfile.TemporaryDirectory(prefix="driftmap-state-") as scratch:
            pinned = f"{results}@{hub.dataset_info(results).sha}"
            root = Path(hub.snapshot_download(pinned, scratch, repo_type="dataset", allow_patterns=STATE_FILES))
            records = [_parsed(RunRecord, path) for path in sorted((root / "runs").glob("*.json"))]
            generated = sorted(path.stem for path in root.glob(f"*/generate-{mode}-*.json"))
            screen = _parsed(AlphaScreen, root / "alpha_screen.json")
        # by run id, so the newest ok record of a kind wins
        ok = {
            record.kind: record
            for record in records
            if record and (record.mode, record.config_hash, record.status) == (mode, settings.config_hash(), "ok")
        }
        return State(
            languages=tuple(settings.languages),
            # chain cannot name the generate target yet, so the strictest one is checked
            checkpoint="drifted",
            merged_sha256=ok["drift"].merged_sha256 if "drift" in ok else None,
            responses_complete="generate" in ok and generated[-1:] == [ok["generate"].run_id],
            repaired_revision=output(ok.get("repair"), settings.repos.repaired),
            gguf_record=output(ok.get("quant"), settings.repos.gguf) is not None,
            recover_status=screen.recover_status if screen else None,
        )

    return read


@app.command()
@_finish
def chain(
    kinds: Annotated[
        list[Kind], typer.Argument(help="Job kinds in run order; generate is checked as the drifted target.")
    ],
    mode: Annotated[Mode, typer.Option(help="full is the paid run.")] = "full",
) -> ExitCode:
    """Launch and follow each kind in turn; stop at a missing input, a refusal or a job that does not end ok."""
    names = [kind.value for kind in kinds]
    settings, deps = _open("chain", kinds=names, mode=mode)
    return launcher.chain(settings, deps, names, mode, state=_state(settings, deps.hub, mode))


@app.command("wait")
@_finish
def wait_job(
    job_id: Annotated[str, typer.Argument(help="A job id, or unsubmitted:<run_id> as the ledger holds it.")],
) -> ExitCode:
    """Follow a submitted job to its end, record its cost and print the next action."""
    # a stale report must never keep a running job off the ledger
    settings, deps = _open("wait", ready=False, job=job_id)
    return launcher.wait_job(settings, deps, job_id)


@ledger_app.command("reconcile")
@_finish
def reconcile() -> ExitCode:
    """Compare billing usage with the ledger; exit 1 when billing is ahead by more than $0.50."""
    _, deps = _open("ledger reconcile", ready=False)
    return launcher.reconcile(deps)


@decide_app.command("drift")
@_finish
def decide_drift(
    note: Annotated[str, typer.Option(help="Why; stored and published with the decision.")],
    proceed: Annotated[bool, typer.Option("--proceed", help="Repair the drift as measured.")] = False,
    extend: Annotated[bool, typer.Option("--extend", help="Run a longer drift first.")] = False,
) -> None:
    """Record the call on a drift below the threshold and push it to the results repo."""
    if proceed == extend:
        raise typer.BadParameter("pass exactly one of --proceed and --extend")
    if not note.strip():
        raise typer.BadParameter("--note needs text")
    decision = "proceed" if proceed else "extend"
    settings, deps = _open("decide drift", decision=decision, note=note)
    leaks = [name for name, pattern in redact.PATTERNS.items() if pattern.search(note)]
    if leaks:
        raise DriftmapError(ExitCode.REDACTION, f"--note holds {', '.join(leaks)}; the decision is published")
    path = deps.repo_root / DECISION_PATH
    record = write_decision(path, decision, note, now=deps.now)
    results = f"{settings.owner}/{settings.repos.results}"
    in_repo = DECISION_PATH.relative_to(LOCAL / "results").as_posix()
    try:
        deps.hub.upload_file(
            results, in_repo, path, repo_type="dataset", commit_message=f"record the drift decision: {decision}"
        )
    except httpx2.HTTPError as error:
        message = f"the decision is stored in {DECISION_PATH} but was not pushed ({type(error).__name__})"
        raise DriftmapError(ExitCode.PUSH_FAILED, f"{message}; run the command again") from error
    typer.echo(f"recorded {decision} at {record.ts:%Y-%m-%dT%H:%M:%SZ} and pushed {in_repo} to {results}")


@app.command()
@_finish
def scan(paths: Annotated[list[Path], typer.Argument(exists=True, help="Files, or directories to walk.")]) -> None:
    """Look for tokens, account ids and home paths in files bound for publication; exit 6 on a finding."""
    files = sorted(
        {file for path in paths for file in ([path] if path.is_file() else path.rglob("*")) if file.is_file()}
    )
    _inputs("scan", paths=paths, files=len(files))
    findings = redact.scan(files, data_root=Path.cwd() / DATA_DIR)
    if findings:
        raise redact.RedactionFinding(findings)
    typer.echo(f"no finding in {len(files)} file(s)")
