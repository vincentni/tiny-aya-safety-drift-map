"""The driftmap command: readiness, data, launches, the ledger, tables, figures, the decisions, status and the scan."""

import argparse
import enum
import fnmatch
import functools
import hashlib
import importlib
import json
import logging
import shlex
import tempfile
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

import httpx2
import typer
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from pydantic import BaseModel, ValidationError

from driftmap.core import ids, redact
from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import (
    AlphaRow,
    AlphaScreen,
    Decision,
    GenerationRow,
    LabelRow,
    LedgerLine,
    Mode,
    Preamble,
    RunRecord,
    Tables,
    dump,
    load,
)
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings, load_settings
from driftmap.jobs import client as jobs_client
from driftmap.jobs import launcher, readiness
from driftmap.jobs.kinds import KINDS, State, prerequisites
from driftmap.jobs.ledger import Ledger
from driftmap.jobs.plan import MASK, SECRET_NAMES
from driftmap.jobs.snapshot import DECISION_PATH, LOCAL, TABLES_PATH, Deps, _billing_used, tables_path, write_decision

log = logging.getLogger(__name__)

CONFIG = "driftmap.toml"
REVISIONS = "revisions.json"
DATA_DIR = LOCAL / "data"
# what the upload commit of another prompt condition's tables adds to its message
CONDITION_SAID = {"strip": ", no safety preamble", "patch": ", safety patch"}
# the prompt conditions the breakdown and the figures cover
CONDITIONS = ids.CONDITIONS
# beside the tables; it covers every prompt condition
BREAKDOWN = "breakdown.json"
# beside the tables: the script variants against their parent prompts, per condition and column
VARIANTS = "variants.json"
# where `judge validate sample` keeps the files of the public sets
VALIDATION_CACHE = LOCAL / "judgeval" / "cache"
FIGURES_DIR = Path("docs") / "figures"
# label set and its folder under prompts/ in the evaluation dataset
PROMPT_SETS = (("harmful", "harmful"), ("benign", "benign_heldout"), ("xstest", "xstest"))
# 128 + SIGINT, as a shell reports it
INTERRUPTED = 130
# job options the launcher writes itself
OWNED = ("--run-id", "--timeout-s", "--config-hash", "--code-sha", "--submitted-at", "--limit", "--force-flush-after-s")
# what a job exits 2 without, and how to type it
JOB_ARGS = {
    "generate": (("--kind",), "--kind baseline|drifted|repaired"),
}
# kinds whose job arguments a launch reads with the script's own parser, before it submits
READ_HERE = ("drift", "generate", "repair", "quant", "quant_eval")
# a quant source as a refusal names it
SOURCE_NAMES = {"repaired": "repaired model", "merged": "merged fine-tuned model"}

Target = Literal["baseline", "drifted", "repaired"]
FineTune = Literal["standard", "aggressive"]
FINE_TUNE_HELP = (
    "The fine-tune in the fine-tuned column. standard: the one trained under the served template. aggressive: the "
    "one trained with no safety preamble, written to files of its own; the two never share a table."
)
MODE_HELP = (
    "smoke: a short rehearsal on a few rows, priced at the smoke timeout and held under the smoke ceiling. "
    "full: the real run, priced at its whole timeout; refused without a passing smoke record of the same job."
)

Kind = enum.StrEnum("Kind", {name: name for name in KINDS})

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Evaluation and training rows.")
variants_app = typer.Typer(no_args_is_help=True, help="The prompts again, romanized or mixed with English.")
ledger_app = typer.Typer(no_args_is_help=True, help="The cost ledger.")
table_app = typer.Typer(no_args_is_help=True, help="The result tables.")
judge_app = typer.Typer(no_args_is_help=True, help="Labels: the closed-model judge and a reparse of stored outputs.")
validate_app = typer.Typer(no_args_is_help=True, help="The judge against human labels of public sets, and itself.")
decide_app = typer.Typer(no_args_is_help=True, help="Calls only the author can make.")
repair_app = typer.Typer(no_args_is_help=True, help="The choice between the screen run and the apply run.")
app.add_typer(data_app, name="data")
data_app.add_typer(variants_app, name="variants")
app.add_typer(ledger_app, name="ledger")
app.add_typer(table_app, name="table")
app.add_typer(judge_app, name="judge")
judge_app.add_typer(validate_app, name="validate")
app.add_typer(decide_app, name="decide")
app.add_typer(repair_app, name="repair")


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
        except KeyboardInterrupt:
            # the launcher has already printed how to recover from a cut submit
            typer.echo("interrupted", err=True)
            for job_id in _pending():
                typer.echo(f"{job_id} is still pending; follow it with: driftmap wait {job_id}", err=True)
            code = INTERRUPTED
        except Exception:  # anything unforeseen leaves as a masked traceback
            typer.echo(redact.mask(traceback.format_exc()), err=True)
            code = ExitCode.UNEXPECTED
        raise typer.Exit(int(code))

    return run


def _pending() -> list[str]:
    """Ids the ledger still holds as pending; none when it cannot be read."""
    try:
        lines = Ledger(Path.cwd() / launcher.LEDGER_PATH).read()
    except (OSError, DriftmapError):
        return []
    return [job_id for job_id, line in lines.items() if line.pending]


def _record[T: BaseModel](cls: type[T], path: Path) -> T | None:
    """A local record, None when absent; one that does not validate is exit 2, named by field and never by value."""
    try:
        return load(cls, path)
    except FileNotFoundError:
        return None
    except ValidationError as error:
        faults = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        raise DriftmapError(ExitCode.USAGE, f"{path.name}: {faults}") from None


def _launch_inputs(root: Path) -> None:
    """Exit 2 on a torn local file a launch reads, before anything is submitted."""
    _record(Decision, root / DECISION_PATH)
    _record(Tables, root / TABLES_PATH)


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
    repin: Annotated[
        bool,
        typer.Option("--repin", help="Move the evaluation pin to this build; run records made so far stop counting."),
    ] = False,
) -> None:
    """Build the evaluation and training rows; unless --dry-run, upload them and pin every revision used."""
    if dry_run and repin:
        raise typer.BadParameter("--repin moves a pin and --dry-run pins nothing; pass one of them")
    # a dry run only reads, and is how the sources are first looked at
    settings, deps = _open("data build", ready=not dry_run, dry_run=dry_run, repin=repin, out=DATA_DIR)
    # pulls the optional data packages
    from driftmap.data import build

    built = build.build(
        settings, deps.hub, deps.repo_root / DATA_DIR, dry_run=dry_run, revisions_path=deps.repo_root / REVISIONS
    )
    for language, counts in built.manifest.per_language.items():
        row = f"{language}  harmful {counts.harmful}  train {counts.benign_train}  held-out {counts.benign_heldout}"
        typer.echo(f"{row}  undersampled" if counts.undersampled else row)
    typer.echo(f"note: {built.manifest.note}")
    if dry_run:
        typer.echo("dry run: nothing was uploaded or pinned")
        return
    target = f"{settings.owner}/{settings.repos.eval}"
    typer.echo(
        f"uploaded to {target}" if built.uploaded else f"{target} already holds these rows; nothing was uploaded"
    )
    pinned = settings.revisions.get(target)
    if pinned == built.commit:
        typer.echo(f"{target} is already pinned at {pinned}")
        return
    stale = "the config hash changes with the pin, so run records made so far stop counting"
    if pinned and not repin:
        if not built.uploaded:
            # the head moved under other writers; the rows are where the pin is
            typer.echo(f"{target} stays pinned at {pinned}")
            return
        message = f"the upload is commit {built.commit}, but {target} is pinned at {pinned}; the pin was not moved"
        raise DriftmapError(ExitCode.GUARD_REFUSAL, f"{message}. To move it: driftmap data build --repin; {stale}")
    path = deps.repo_root / REVISIONS
    on_file = _record(readiness.Revisions, path)
    pins = {**(on_file.root if on_file else {}), target: built.commit}
    dump(readiness.Revisions(dict(sorted(pins.items()))), path)
    moved = f"; it was {pinned}: {stale}" if pinned else ""
    typer.echo(f"pinned {target} at {built.commit}{moved}")
    typer.echo(f"commit and push {REVISIONS}, then run `driftmap ready` again")


VARIANT_SET_HELP = "The set of the parent prompts: harmful or benign."
VARIANT_CODE_HELP = "The variant, such as hi-Latn or es-mixed."


def _variant_lines(files: Iterable[Any]) -> None:
    """One line per set and variant: its rows, the parents left without one, and where its text came from."""
    for made in files:
        typer.echo(f"{made.set:<9}{made.code:<10}{len(made.rows):>5} rows{made.left_out:>5} left out  {made.method}")


@variants_app.command("draft")
@_finish
def variants_draft(
    set_: Annotated[str, typer.Option("--set", help=VARIANT_SET_HELP)],
    code: Annotated[str, typer.Option(help=VARIANT_CODE_HELP)],
    out: Annotated[Path | None, typer.Option(help="The requests file; under the data folder when not given.")] = None,
) -> None:
    """Write one rewriting request per parent prompt of a variant, for rewriting elsewhere; no model is called."""
    _inputs("data variants draft", set=set_, code=code, out=out)
    from driftmap.data import variants

    root = Path.cwd()
    variants.spec(set_, code)
    path, count = variants.draft(root / DATA_DIR, set_, code, out)
    typer.echo(f"{count} requests in {path.relative_to(root) if path.is_relative_to(root) else path}")


@variants_app.command("accept")
@_finish
def variants_accept(
    set_: Annotated[str, typer.Option("--set", help=VARIANT_SET_HELP)],
    code: Annotated[str, typer.Option(help=VARIANT_CODE_HELP)],
    source: Annotated[Path, typer.Option(help="The returned file: JSON lines with parent_prompt_id and prompt.")],
    partial: Annotated[
        bool, typer.Option("--partial", help="Leave out the rows at fault and the parents without a rewrite.")
    ] = False,
) -> None:
    """Check a returned file of rewritten prompts and store its rewrites for the build; exit 12 on a row at fault."""
    _inputs("data variants accept", set=set_, code=code, source=source, partial=partial)
    from driftmap.data import variants

    variants.spec(set_, code)
    found = variants.accept(Path.cwd() / DATA_DIR, set_, code, source, partial=partial)
    typer.echo(f"kept {found.kept}, left out {len(found.dropped)}, without a rewrite {found.missing}")
    for fault in found.dropped:
        typer.echo(redact.mask(f"left out: {fault}"))


@variants_app.command("build")
@_finish
def variants_build() -> None:
    """Build the variant files from the transliteration libraries and the accepted rewrites; uploads nothing."""
    root = Path.cwd()
    settings = load_settings(root / CONFIG)
    _inputs("data variants build", out=DATA_DIR / "variants", config=f"{CONFIG} ({settings.config_hash()[:12]})")
    from driftmap.data import variants

    _variant_lines(variants.build(settings, root / DATA_DIR))
    typer.echo("nothing was uploaded; the upload builds again and screens the rows with the pinned model's tokenizer")


@variants_app.command("upload")
@_finish
def variants_upload() -> None:
    """Build the variants, screened by the pinned model's tokenizer, and add them to the evaluation dataset's head.

    Prints the commit a generate run over the variants names; no pin moves.
    """
    settings, deps = _open("data variants upload", data=DATA_DIR)
    from driftmap.data import variants

    try:
        done = variants.upload(settings, deps.hub, deps.repo_root / DATA_DIR)
    except httpx2.HTTPError as error:
        message = f"the variants were not uploaded ({type(error).__name__})"
        raise DriftmapError(ExitCode.PUSH_FAILED, f"{message}; run the command again") from error
    _variant_lines(done.files)
    written = [made for made in done.files if made.rows]
    rows, files = sum(len(made.rows) for made in written), f"{len(written)} file{'s' * (len(written) != 1)}"
    target = f"{settings.owner}/{settings.repos.eval}"
    typer.echo(f"uploaded {rows} variant prompts in {files} to {target} at commit {done.commit}")
    launch = f"driftmap launch generate --wait -- --kind baseline --sets variants --variants-rev {done.commit}"
    typer.echo(f"{'next':<10}{launch}")


@app.command()
@_finish
def plan(kind: Kind, mode: Annotated[Mode, typer.Option(help=MODE_HELP)] = "smoke") -> ExitCode:
    """Print the resolved job, its cost at timeout, the spend so far and every guard verdict; submit nothing."""
    settings, deps = _open("plan", kind=kind.value, mode=mode)
    _launch_inputs(deps.repo_root)
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


def _job_args(kind: str, mode: str, given: list[str] | None) -> list[str]:
    """The arguments passed on to the job script; exit 2 for ones the job or the launcher cannot take."""
    given = given or []
    flags = [arg.split("=", 1)[0] for arg in given if arg.startswith("--")]
    # argparse takes any unambiguous prefix of an option
    owned = [flag for flag in flags if len(flag) > 2 and any(name.startswith(flag) for name in OWNED)]
    if owned:
        raise typer.BadParameter(f"{owned[0]} is set by the launcher; a row cap goes before `--` as --limit N")
    if any(redact.mask(arg) != arg for arg in given):
        raise typer.BadParameter(
            "a job argument holds a token, a home path or a control sequence; it is printed and stored"
        )
    needed, form = JOB_ARGS.get(kind, ((), ""))
    if any(flag not in flags for flag in needed):
        raise typer.BadParameter(f"{kind} needs its job arguments: driftmap launch {kind} --mode {mode} -- {form}")
    return given


def _read_job_args(kind: str, args: Sequence[str]) -> argparse.Namespace | None:
    """Job arguments as the job script will read them, None for a kind not read here; exit 2 in the parser's words."""
    if kind not in READ_HERE:
        return None
    # a script pulls in the helpers of its job, so it is loaded for its own launch only
    from driftmap import scripts

    parser = scripts.passed_on_parser(importlib.import_module(KINDS[kind].module)._arguments)

    def refuse(message: str) -> NoReturn:
        raise typer.BadParameter(f"the {kind} job would stop on its arguments: {message}")

    parser.error = refuse
    return parser.parse_args(list(args))


def _apply_alpha(parsed: argparse.Namespace, settings: Settings, mode: str) -> float | None:
    """The alpha a repair launch asks to push, None when it screens; exit 2 for alphas the job cannot take."""
    from driftmap.scripts import repair

    try:
        alphas = repair.asked_alphas(parsed, settings, mode)
    except DriftmapError as error:
        raise DriftmapError(ExitCode.USAGE, f"the repair job would stop on its arguments: {error}") from None
    return alphas[0] if parsed.apply is not None else None


@app.command()
@_finish
def launch(
    kind: Kind,
    mode: Annotated[Mode, typer.Option(help=MODE_HELP)] = "smoke",
    override_smoke: Annotated[
        bool,
        typer.Option("--override-smoke", help="Submit a full run with no passing smoke record of it; needs --reason."),
    ] = False,
    override_soft: Annotated[
        bool,
        typer.Option(
            "--override-soft",
            help="Check spend against the hard ceiling plus own money, not the soft ceiling, and launch even when "
            "billing is unreadable and the ledger is empty; needs --reason.",
        ),
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
    # masked before folded: a line break can cut a token in two
    reason = " ".join("".join(char if char.isprintable() else " " for char in redact.mask(reason)).split())
    if overrides and not reason:
        raise typer.BadParameter(
            f"{' and '.join(overrides)}: an override needs --reason TEXT, kept in the ledger notes"
        )
    if reason and not overrides:
        raise typer.BadParameter("--reason goes with --override-smoke or --override-soft")
    args = _job_args(kind.value, mode, job_args)
    parsed = _read_job_args(kind.value, args)
    rows = {"kind": kind.value, "mode": mode, "dry_run": dry_run, "wait": wait, "limit": limit}
    settings, deps = _open("launch", **rows, job_args=args, overrides=overrides, reason=reason or None)
    _launch_inputs(deps.repo_root)
    missing, named = [], None
    if kind.value == "repair":
        apply = _apply_alpha(parsed, settings, mode)
        from driftmap.scripts import repair

        preamble = repair.condition(parsed)
        chosen = {"fine_tune": parsed.fine_tune, "preamble": preamble}
        missing = prerequisites("repair", _state(settings, deps, mode, None, apply=apply, **chosen)("repair"))
        if parsed.fine_tune != "standard":
            # naming the fine-tune stands for the drift decision, so its own tables must show it measured
            named = (parsed.fine_tune, preamble)
            held = tables_path(preamble, parsed.fine_tune)
            table = _record(Tables, deps.repo_root / held)
            if table is None or "drifted" not in table.columns or table.columns["drifted"].state != "tabled":
                build = f"driftmap table build --preamble {preamble} --fine-tune {parsed.fine_tune}"
                missing.append(f"{held.name} with the fine-tuned column tabled; run `{build}`")
    elif kind.value in ("quant", "quant_eval"):
        missing = _quant_missing(kind.value, parsed, settings, deps.hub, mode)
    elif kind.value == "generate" and parsed.kind == "repaired":
        missing = _repaired_missing(parsed, settings, deps.hub, mode)
    if missing:
        typer.echo(f"{kind.value} cannot run yet; missing: {'; '.join(missing)}")
        if not dry_run:
            return ExitCode.GUARD_REFUSAL
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
        args=args,
        **({"named": named} if named else {}),
    )


def _parsed[T: BaseModel](cls: type[T], path: Path) -> T | None:
    try:
        return load(cls, path)
    except FileNotFoundError:
        return None
    except ValidationError:
        log.warning("%s does not parse as %s; it is ignored", path.name, cls.__name__)
        return None


def _unreadable(repo: str, error: Exception) -> DriftmapError:
    return DriftmapError(ExitCode.UNEXPECTED, f"{repo} could not be read ({type(error).__name__})")


def _head(hub: Hub, repo: str) -> Any:
    """A dataset's head, with the names of its files; exit 1 naming the repo that cannot be read."""
    try:
        return hub.dataset_info(repo)
    except (httpx2.HTTPError, LocalEntryNotFoundError) as error:
        raise _unreadable(repo, error) from error


def _files(hub: Hub, repo: str, revision: str, scratch: str, patterns: list[str]) -> Path:
    """Files of a dataset at ``revision`` in ``scratch``; exit 1 naming a repo it cannot read."""
    try:
        return Path(hub.snapshot_download(f"{repo}@{revision}", scratch, repo_type="dataset", allow_patterns=patterns))
    except (httpx2.HTTPError, LocalEntryNotFoundError) as error:
        raise _unreadable(repo, error) from error


def _snapshot(hub: Hub, repo: str, scratch: str, patterns: list[str], head: Any = None) -> Path:
    """Files of a dataset at ``head``, by default its head now, in ``scratch``; exit 1 naming a repo it cannot read."""
    return _files(hub, repo, (head or _head(hub, repo)).sha, scratch, patterns)


def _results(
    settings: Settings, hub: Hub, mode: str = "full", preamble: str = "keep", fine_tune: str = "standard"
) -> tuple[list[RunRecord], AlphaScreen | None]:
    """Run records, by run id, and an alpha screen at the results repo's head; a torn file is left out.

    The screen is the one of ``mode``, of the fine-tune and of the prompt condition it was screened under.
    """
    from driftmap.repair.screen_gen import screen_path

    stored = screen_path(mode, preamble, fine_tune)
    with tempfile.TemporaryDirectory(prefix="driftmap-results-") as scratch:
        root = _snapshot(hub, f"{settings.owner}/{settings.repos.results}", scratch, ["runs/*.json", stored])
        records = [_parsed(RunRecord, path) for path in sorted((root / "runs").glob("*.json"))]
        return [record for record in records if record], _parsed(AlphaScreen, root / stored)


def _drifted(
    records: Iterable[RunRecord], settings: Settings, mode: str, fine_tune: str = "standard"
) -> tuple[RepoRef, str] | None:
    """The merged checkpoint of the latest ok drift run of ``mode`` and its weights hash, as the repair job picks it.

    ``fine_tune`` names the fine-tune whose merged repo is meant.
    """
    repo = f"{settings.owner}/{settings.repos.merged}{ids.repo_suffix(mode, ids.FINE_TUNES[fine_tune])}"
    for run in reversed(sorted(records, key=lambda run: run.started)):
        pushed = next((ref for ref in run.outputs if ref.repo == repo), None)
        if (run.kind, run.status, run.mode) == ("drift", "ok", mode) and run.merged_sha256 and pushed:
            return RepoRef(repo, pushed.revision), run.merged_sha256
    return None


def _repaired_missing(parsed: argparse.Namespace, settings: Settings, hub: Hub, mode: str) -> list[str]:
    """What a generate job over a repaired model would stop on: its repo, or the revision ``--checkpoint`` names.

    Read as the job reads it: the standard fine-tune's repaired repo of ``mode`` unless the other one's is named.
    """
    repos = [
        f"{settings.owner}/{settings.repos.repaired}{ids.repo_suffix(mode, preamble)}" for preamble in ids.PREAMBLES
    ]
    given = parsed.checkpoint
    if given is not None and given.repo not in repos:
        return [f"--checkpoint naming a repaired model of {mode} mode: {' or '.join(repos)}"]
    repo, revision = (given.repo, given.revision) if given is not None else (repos[0], None)

    def head(name: str) -> str | None:
        try:
            return hub.model_info(RepoRef(name)).sha
        except RepositoryNotFoundError:
            return None

    try:
        hub.model_info(RepoRef(repo, revision))
    except RevisionNotFoundError:
        return [f"revision {revision[:12]} of {repo}; its head is `--checkpoint {repo}@{head(repo)}`"]
    except RepositoryNotFoundError:
        lacking = f"{repo}, which no repair run has pushed"
        # an unnamed launch reads the standard fine-tune's repair; the other one's is read only by name
        there = [(other, head(other)) for other in repos if other != repo and given is None]
        other, sha = next(((other, sha) for other, sha in there if sha), (None, None))
        form = f"; the repaired model on the Hub is the other fine-tune's: add `--checkpoint {other}@{sha}`"
        return [lacking + (form if other else "")]
    except (httpx2.HTTPError, LocalEntryNotFoundError) as error:
        raise _unreadable(repo, error) from error
    return []


def _quant_missing(kind: str, parsed: argparse.Namespace, settings: Settings, hub: Hub, mode: str) -> list[str]:
    """The record a quant or quant-eval job of this source will look for and not find; empty when it is there.

    Read as the job reads it: the ok runs of ``mode``, whatever config they ran under.
    """
    from driftmap import quant

    # the repaired model is read at its head; a file named on the command line, or the one a resumed run read,
    # is the job's to check
    named = parsed.source != "merged" if kind == "quant" else parsed.gguf is not None or parsed.resume
    if named:
        return []
    records, _ = _results(settings, hub, mode)
    done = [run for run in records if (run.status, run.mode) == ("ok", mode)]
    checkpoint, gguf = quant.names(settings, mode, parsed.source)

    def pushed(made: str, repo: str) -> list[str]:
        return [ref.revision for run in done if run.kind == made for ref in run.outputs if ref.repo == repo]

    if kind == "quant":
        hashed = [run for run in done if run.kind == "drift" and run.merged_sha256]
        revisions = [ref.revision for run in hashed for ref in run.outputs if ref.repo == checkpoint]
        asked = parsed.checkpoint.revision if parsed.checkpoint else None
        if asked in revisions if asked else revisions:
            return []
        return [f"drift record with merged_sha256 for {checkpoint}{f' at {asked[:12]}' if asked else ''}"]
    if pushed("quant", gguf):
        return []
    others = [source for source in quant.SOURCES if pushed("quant", quant.names(settings, mode, source)[1])]
    other = [f"the finished quant run is of the {SOURCE_NAMES[made]}: add `-- --source {made}`" for made in others]
    return ["; ".join([f"a finished quant run that pushed {gguf}", *other[:1]])]


def _benign_labelled(
    settings: Settings, hub: Hub, checkpoint: RepoRef, mode: str, preamble: str = "keep"
) -> frozenset[str]:
    """Languages whose benign label rows of ``checkpoint`` under ``preamble`` are in the evaluation dataset."""
    slug = ids.slug(checkpoint)
    with tempfile.TemporaryDirectory(prefix="driftmap-labels-") as scratch:
        wanted = [ids.labels_path(slug, "*", mode, benign=True, preamble=preamble)]
        root = _snapshot(hub, f"{settings.owner}/{settings.repos.eval}", scratch, wanted)
        paths = {
            code: root / ids.labels_path(slug, code, mode, benign=True, preamble=preamble)
            for code in settings.languages
        }
        return frozenset(code for code, path in paths.items() if path.is_file())


def _state(
    settings: Settings,
    deps: Deps,
    mode: str,
    target: str | None,
    *,
    apply: float | None = None,
    fine_tune: str = "standard",
    preamble: str = "keep",
) -> Callable[[str], State]:
    """Reader of what the Hub holds before a step is submitted; ``apply`` is the alpha the repair step is to push.

    ``fine_tune`` and ``preamble`` name the fine-tune a repair step works on and its prompt condition.
    """
    arm = ids.repo_suffix(mode, ids.FINE_TUNES[fine_tune])

    def output(record: RunRecord | None, repo: str) -> str | None:
        refs = record.outputs if record else []
        return next((ref.revision for ref in refs if ref.repo == f"{settings.owner}/{repo}"), None)

    def read(kind: str) -> State:
        records, screen = _results(settings, deps.hub, mode, preamble, fine_tune)
        # only ok runs of this mode under this config count
        runs = {
            record.run_id: record
            for record in records
            if (record.mode, record.config_hash, record.status) == (mode, settings.config_hash(), "ok")
        }
        # by run id, so the newest ok record of a kind wins
        ok = {record.kind: record for record in runs.values()}
        merged = ok["drift"].merged_sha256 if "drift" in ok else None
        benign: frozenset[str] = frozenset()
        if kind == "repair":
            # the job takes its drift run and the stored screen whatever config they were made under
            drifted, merged = _drifted(records, settings, mode, fine_tune) or (None, None)
            benign = _benign_labelled(settings, deps.hub, drifted, mode, preamble) if drifted else benign
        return State(
            languages=tuple(settings.languages),
            checkpoint=target or "drifted",
            apply=apply,
            merged_sha256=merged,
            benign_labels=benign,
            repaired_revision=output(ok.get("repair"), settings.repos.repaired + arm),
            gguf_record=output(ok.get("quant"), settings.repos.gguf + ids.repo_suffix(mode)) is not None,
            recover_status=screen.recover_status if screen and (kind == "repair" or "repair" in ok) else None,
            chosen=screen.chosen if screen and kind == "repair" else None,
        )

    return read


@app.command()
@_finish
def chain(
    kinds: Annotated[list[Kind], typer.Argument(help="Job kinds in run order.")],
    mode: Annotated[Mode, typer.Option(help=MODE_HELP)] = "full",
    target: Annotated[Target | None, typer.Option(help="The checkpoint the generate step reads.")] = None,
) -> ExitCode:
    """Launch and follow each kind in turn; stop at a missing input, a refusal or a job that does not end ok."""
    names = [kind.value for kind in kinds]
    if names.count("generate") > 1:
        raise typer.BadParameter("one generate step per chain: its target is one checkpoint")
    if ("generate" in names) != (target is not None):
        raise typer.BadParameter(
            "--target names the generate step: chain generate ... --target baseline|drifted|repaired"
        )
    if "repair" in names:
        raise typer.BadParameter(
            "repair is not one chain step: `driftmap launch repair --wait` screens the alphas, `driftmap repair "
            "choose` decides from their labels, and `driftmap launch repair --wait -- --apply ALPHA` pushes the model"
        )
    settings, deps = _open("chain", kinds=names, mode=mode, target=target)
    _launch_inputs(deps.repo_root)
    job_args = {"generate": ["--kind", target]} if target else {}
    state = _state(settings, deps, mode, target)
    return launcher.chain(settings, deps, names, mode, state=state, args=job_args)


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


def _jsonl[T](path: Path, pick: Callable[[dict[str, Any]], T]) -> list[T]:
    """One value per row of a JSONL file; exit 12 naming the file and line of a row of another shape."""
    found = []
    # split on newlines only; a prompt may hold other line separators
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        try:
            found += [pick(json.loads(line))] if line.strip() else []
        except (ValueError, LookupError, TypeError):
            where = f"{path.parent.name}/{path.name}:{number}"
            raise DriftmapError(ExitCode.INTEGRITY, f"{where}: not a row of the shape its file holds") from None
    return found


def _targets(data: Path, labels_dir: Path, settings: Settings, mode: str) -> dict[str, dict[str, list[str]]]:
    """Prompt ids per set and language as the evaluation dataset lists them; a smoke table takes the rows judged."""
    targets = {
        name: {
            path.stem: _jsonl(path, lambda row: row["prompt_id"])
            for path in (data / "prompts" / folder).glob("*.jsonl")
        }
        for name, folder in PROMPT_SETS
    }
    if mode == "smoke":
        # a smoke run takes the first rows of each file; the longest column says how many
        sizes: dict[tuple[str, str], int] = {}
        for path in [*labels_dir.glob("*/*.jsonl"), *labels_dir.glob("*/benign/*.jsonl")]:
            rows = _jsonl(path, lambda row: (row["set"], row["language"]))
            for key in set(rows):
                sizes[key] = max(sizes.get(key, 0), rows.count(key))
        targets = {
            name: {language: listed[: sizes.get((name, language), 0)] for language, listed in by_language.items()}
            for name, by_language in targets.items()
        }
    benign = settings.sources.benign
    pinned = settings.revisions.get(benign)
    if pinned is None:
        raise DriftmapError(
            ExitCode.USAGE, f"{benign} is not pinned; training prompts take their ids from its revision"
        )
    # training rows carry no id: it is their source text under the pinned revision
    targets["train"] = {
        path.stem: _jsonl(path, lambda row: ids.prompt_id(benign, pinned, row["prompt"][-1]["content"]))
        for path in (data / "sft_train").glob("*.jsonl")
    }
    return targets


# run kinds whose record names the checkpoint a column's labels belong to
LABELLED = ("generate", "quant_eval", "steer")
# what follows the slug of the steered weights in a steered checkpoint's folder: layer and strength
STEERED = "--steer-*"


def _label_folders(
    records: list[RunRecord], settings: Settings, mode: str, table: Any, preamble: str, fine_tune: str = "standard"
) -> tuple[list[str], dict[str, str]]:
    """Label folders the table reads, as patterns, and for each folder an earlier run named the one that replaced it.

    Per column the latest ok run of this mode and condition decides. A generate or quant-eval run names the
    checkpoint it read; a steer run names the weights it steered, whose labels sit under their slug with a steer suffix.
    A run over a checkpoint of the other fine-tune decides nothing here.
    """
    latest: dict[tuple[str, bool], str] = {}
    named: dict[str, tuple[str, bool]] = {}
    for run in sorted(records, key=lambda run: run.started):
        models = [f"{ref.repo}@{ref.revision}" for ref in run.inputs if ref.kind == "model"]
        same_tree = (run.mode, run.status, ids.preamble_of(run.notes)) == (mode, "ok", preamble)
        # a run over the script variants leaves its labels where no table reads
        if run.kind not in LABELLED or not same_tree or not models or ids.variants_of(run.notes):
            continue
        columns = [table.column_of(reference, settings, mode, fine_tune) for reference in models]
        # a quant-eval run also records the checkpoint its file was converted from: the later column is its own
        read = max(range(len(models)), key=lambda at: table.COLUMNS.index(columns[at]) if columns[at] else -1)
        if columns[read] is None and table.column_of(models[read], settings, mode):
            continue
        steered = run.kind == "steer"
        folder = ids.slug(models[read]) + (STEERED if steered else "")
        key = (columns[read] or models[read], steered)
        latest[key], named[folder] = folder, key
    return sorted(latest.values()), {folder: latest[key] for folder, key in named.items() if latest[key] != folder}


def _held(head: Any, prefix: str) -> list[str]:
    """Names of the folders right under ``prefix`` at a dataset's head."""
    inside = (name[len(prefix) :] for sibling in head.siblings or () if (name := sibling.rfilename).startswith(prefix))
    return sorted({rest.split("/")[0] for rest in inside if "/" in rest})


def _left(
    held: list[str],
    read: list[str],
    replaced: dict[str, str],
    table: Any,
    settings: Settings,
    mode: str,
    fine_tune: str = "standard",
) -> dict[str, str]:
    """Why each label folder on the Hub that the table does not read is left."""

    def among(folder: str, patterns: list[str] | dict[str, str]) -> str | None:
        return next((pattern for pattern in patterns if fnmatch.fnmatchcase(folder, pattern)), None)

    left = {}
    for folder in held:
        if among(folder, read):
            continue
        earlier = among(folder, replaced)
        if earlier:
            left[folder] = f"a later ok run read {replaced[earlier]}"
        elif table.column_of(folder, settings, mode) is None:
            left[folder] = "not a checkpoint of this table"
        elif table.column_of(folder, settings, mode, fine_tune) is None:
            left[folder] = f"a checkpoint of the other fine-tune, not the {fine_tune} one"
        else:
            left[folder] = f"no ok {mode} run on record read it"
    return left


def _of_fine_tune(name: str, fine_tune: str) -> str:
    """A file or folder of ``fine_tune``: the standard one's as given, another's with the fine-tune after its stem."""
    path = Path(name)
    return name if fine_tune == "standard" else str(path.with_name(f"{path.stem}-{fine_tune}{path.suffix}"))


def _tables_name(preamble: str, fine_tune: str = "standard") -> str:
    """File of a condition's tables: tables.json, or tables-<tree>[-<fine-tune>].json beside it."""
    return tables_path(preamble, fine_tune).name


def _read_folders(
    records: list[RunRecord],
    settings: Settings,
    mode: str,
    table: Any,
    preamble: str,
    head: Any,
    fine_tune: str = "standard",
) -> list[str]:
    """The label folders a condition's table reads, as patterns; says which label folders on the Hub it leaves."""
    tree = ids.tree(mode, preamble)
    # labels of a revision no finished run generated would put two checkpoints in one column
    folders, replaced = _label_folders(records, settings, mode, table, preamble, fine_tune)
    left = _left(_held(head, f"{tree}labels/"), folders, replaced, table, settings, mode, fine_tune)
    for folder, reason in left.items():
        typer.echo(redact.mask(f"left unread: {tree}labels/{folder}: {reason}"))
    return folders


def _show(path: Path, shown: str, root: Path) -> list[Path]:
    """Write the markdown beside its record, scan both for what must not be published, then print the markdown."""
    files = [path, path.with_suffix(".md")]
    files[1].write_text(shown, encoding="utf-8")
    findings = redact.scan(files)
    if findings:
        raise redact.RedactionFinding(findings)
    typer.echo(redact.mask(shown))
    typer.echo(f"written to {path.parent.relative_to(root)}")
    return files


def _upload(deps: Deps, results: str, in_repo: str, files: list[Path], message: str) -> None:
    """Put the files of one folder under ``in_repo`` in the results repo; exit 7 when the Hub does not take them."""
    folder = files[0].parent
    names = [file.name for file in files]
    try:
        deps.hub.upload_folder(
            results, folder, repo_type="dataset", path_in_repo=in_repo, allow_patterns=names, commit_message=message
        )
    except httpx2.HTTPError as error:
        kept = f"{' and '.join(names)} are in {folder.relative_to(deps.repo_root)} but were not uploaded"
        raise DriftmapError(ExitCode.PUSH_FAILED, f"{kept} ({type(error).__name__}); run the command again") from error
    typer.echo(f"uploaded to {results} under {in_repo}/")


def _flag_on_file(path: Path, hub: Hub, results: str, in_repo: str, flag: str) -> bool:
    """Whether the stored table raised ``flag``; a missing or torn local file gives way to a Hub copy that parses."""
    try:
        if path.exists():
            return bool(load(Tables, path).flags.get(flag))
    except ValidationError:
        path.unlink()
    # the upload of an earlier build: a flag it holds must outlive the local file
    with tempfile.TemporaryDirectory(prefix="driftmap-tables-") as scratch:
        copy = _parsed(Tables, _snapshot(hub, results, scratch, [in_repo]) / in_repo)
    if copy is None:
        return False
    dump(copy, path)
    return bool(copy.flags.get(flag))


@table_app.command("build")
@_finish
def table_build(
    mode: Annotated[
        Mode, typer.Option(help="full: the result tables. smoke: the rehearsal's, kept in their own tree.")
    ] = "full",
    preamble: Annotated[
        Preamble,
        typer.Option(
            help="strip: runs with no safety preamble, written to tables-nopreamble. "
            "patch: runs under the safety patch, written to tables-patched."
        ),
    ] = "keep",
    fine_tune: Annotated[FineTune, typer.Option(help=FINE_TUNE_HELP)] = "standard",
) -> None:
    """Build the tables from the labels on the Hub, print and upload them; exit 8 on a new, undecided drift flag."""
    chosen = {"preamble": preamble} if preamble != "keep" else {}
    chosen |= {"fine_tune": fine_tune} if fine_tune != "standard" else {}
    settings, deps = _open("table build", mode=mode, **chosen)
    # pulls the statistics packages
    from driftmap.table import build as table
    from driftmap.table import render

    root, tree = deps.repo_root, ids.tree(mode, preamble)
    # the repair screen of this fine-tune, under the condition it was screened in
    records, screen = _results(settings, deps.hub, mode, ids.FINE_TUNES[fine_tune], fine_tune)
    results, in_repo = f"{settings.owner}/{settings.repos.results}", f"{tree}tables"
    evaluation = f"{settings.owner}/{settings.repos.eval}"
    head = _head(deps.hub, evaluation)
    folders = _read_folders(records, settings, mode, table, preamble, head, fine_tune)
    labels = [f"{tree}labels/{folder}/*" for folder in folders]
    with tempfile.TemporaryDirectory(prefix="driftmap-table-") as scratch:
        wanted = [*labels, "prompts/*/*.jsonl", "sft_train/*.jsonl"]
        data = _snapshot(deps.hub, evaluation, scratch, wanted, head)
        labels_dir = data / tree / "labels"
        targets = _targets(data, labels_dir, settings, mode)
        known = [*records, *([screen] if screen else [])]
        tables = table.build(labels_dir, targets, known, settings, mode=mode, preamble=preamble, fine_tune=fine_tune)
    flagged = None
    if mode == "smoke" or preamble != "keep" or fine_tune != "standard":
        # only the full table of the standard fine-tune under the served template gates the repair
        folder = root / LOCAL / "results" / ("smoke" if mode == "smoke" else "") / TABLES_PATH.parent.name
        path = folder / _tables_name(preamble, fine_tune)
        dump(tables, path)
    else:
        path, decision = root / TABLES_PATH, _record(Decision, root / DECISION_PATH)
        raised = _flag_on_file(path, deps.hub, results, f"{in_repo}/{path.name}", table.DRIFT_FLAG)
        if raised and table.DRIFT_FLAG not in tables.flags and decision is None:
            # a flag that can no longer be evaluated still gates the repair
            advice = f"{path.name} is left as it is; record a decision with `driftmap decide drift`, then build again"
            raise DriftmapError(
                ExitCode.DRIFT_BELOW_THRESHOLD, f"the drift flag on file cannot be evaluated now; {advice}"
            )
        try:
            table.save(tables, root, {"drift": decision} if decision else {})
        except DriftmapError as error:
            if error.exit_code is not ExitCode.DRIFT_BELOW_THRESHOLD:
                raise
            # said now: a failed upload below must not lose it; the table is shown and uploaded first
            typer.echo(f"drift flag: {error}", err=True)
            flagged = error
    files = _show(path, render.render(tables, settings), root)
    message = f"add {mode} tables{CONDITION_SAID.get(preamble, '')}"
    _upload(deps, results, in_repo, files, message + (f", {fine_tune} fine-tune" if fine_tune != "standard" else ""))
    if flagged:
        raise flagged


def _tagged(data: Path, breakdown: Any) -> dict[str, dict[tuple[str, str], Any]]:
    """Tags of the prompts of every set the breakdown covers, by set, then language and prompt id."""
    folders = dict(PROMPT_SETS)
    return {
        name: {
            (path.stem, prompt_id): tags
            for path in sorted((data / "prompts" / folders.get(name, name)).glob("*.jsonl"))
            for prompt_id, tags in _jsonl(path, breakdown.tags)
        }
        for name in breakdown.SETS
    }


@table_app.command("breakdown")
@_finish
def table_breakdown(
    mode: Annotated[
        Mode, typer.Option(help="full: the result labels. smoke: the rehearsal's, kept in their own tree.")
    ] = "full",
    fine_tune: Annotated[FineTune, typer.Option(help=FINE_TUNE_HELP)] = "standard",
) -> None:
    """Break the harmful-response rate down by harm category, language, prompt origin and answer length.

    Covers every prompt condition that has labels, and each prompt's outcome before and after fine-tuning; whether
    an answer was cut off at the token limit is read from the stored responses. Prints and uploads the result.
    """
    # nothing is launched or billed to the ledger, so no readiness report is needed
    chosen = {"fine_tune": fine_tune} if fine_tune != "standard" else {}
    settings, deps = _open("table breakdown", ready=False, mode=mode, **chosen)
    # pulls the statistics packages
    from driftmap.table import breakdown
    from driftmap.table import build as table

    # the repair screen of this fine-tune, under the condition it was screened in
    records, screen = _results(settings, deps.hub, mode, ids.FINE_TUNES[fine_tune], fine_tune)
    results, evaluation, stored = (
        f"{settings.owner}/{name}" for name in (settings.repos.results, settings.repos.eval, settings.repos.responses)
    )
    head = _head(deps.hub, evaluation)
    folders = dict(PROMPT_SETS)
    wanted, answers = [f"prompts/{folders.get(name, name)}/*.jsonl" for name in breakdown.SETS], []
    for preamble in CONDITIONS:
        for folder in _read_folders(records, settings, mode, table, preamble, head, fine_tune):
            wanted.append(f"{ids.tree(mode, preamble)}labels/{folder}/*")
            answers += [ids.responses_path(folder, name, "*", mode, preamble) for name in breakdown.SETS]
    with tempfile.TemporaryDirectory(prefix="driftmap-breakdown-") as scratch:
        data = _snapshot(deps.hub, evaluation, f"{scratch}/labels", wanted, head)
        # without a label folder there is no answer to fetch, and an empty pattern list must not fetch them all
        said = _snapshot(deps.hub, stored, f"{scratch}/responses", answers) if answers else Path(scratch)
        labels = {preamble: data / ids.tree(mode, preamble) / "labels" for preamble in CONDITIONS}
        responses = {preamble: said / ids.tree(mode, preamble) / "responses" for preamble in CONDITIONS}
        known = [*records, *([screen] if screen else [])]
        options = {"mode": mode, "responses": responses, "fine_tune": fine_tune}
        built = breakdown.build(labels, _tagged(data, breakdown), known, settings, **options)
    root = deps.repo_root
    folder = root / LOCAL / "results" / ("smoke" if mode == "smoke" else "") / TABLES_PATH.parent.name
    path = folder / _of_fine_tune(BREAKDOWN, fine_tune)
    dump(built, path)
    files = _show(path, breakdown.render(built), root)
    message = f"add {mode} breakdown" + (f", {fine_tune} fine-tune" if fine_tune != "standard" else "")
    _upload(deps, results, f"{ids.tree(mode)}tables", files, message)


@table_app.command("variants")
@_finish
def table_variants(
    mode: Annotated[
        Mode, typer.Option(help="full: the result labels. smoke: the rehearsal's, kept in their own tree.")
    ] = "full",
    fine_tune: Annotated[FineTune, typer.Option(help=FINE_TUNE_HELP)] = "standard",
) -> None:
    """Compare the script variants with their parent prompts, per condition and model; print and upload it.

    Covers each condition with a finished variants run; leaves out a variant judged another request than its parent.
    The stored benign answers are the control: how often they are empty or cut off, and how long they are.
    """
    # nothing is launched or billed to the ledger, so no readiness report is needed
    chosen = {"fine_tune": fine_tune} if fine_tune != "standard" else {}
    settings, deps = _open("table variants", ready=False, mode=mode, **chosen)
    # pulls the statistics packages
    from driftmap.table import variants

    records, _ = _results(settings, deps.hub, mode)
    results, evaluation, stored = (
        f"{settings.owner}/{name}" for name in (settings.repos.results, settings.repos.eval, settings.repos.responses)
    )
    found = {name: variants.sources(records, settings, mode, name, fine_tune) for name in ids.CONDITIONS}
    read = [(condition, source) for condition, columns in found.items() for source in columns.values()]
    if not read:
        raise DriftmapError(ExitCode.USAGE, f"no finished {mode} run over the script variants is on record")
    # per run the files of its checkpoint, for the parent prompts and then for the variants: labels and benign answers
    trees = [(ids.slug(source.checkpoint), condition, among) for condition, source in read for among in (False, True)]
    labels = [ids.labels_path(slug, "*", mode, preamble=condition, variants=among) for slug, condition, among in trees]
    answers = [ids.responses_path(slug, "benign", "*", mode, condition, among) for slug, condition, among in trees]
    with tempfile.TemporaryDirectory(prefix="driftmap-variants-") as scratch:
        data = _snapshot(deps.hub, evaluation, f"{scratch}/labels", labels)
        said = _snapshot(deps.hub, stored, f"{scratch}/responses", answers)
        links = {}
        for commit in sorted({source.variants_revision for _, source in read}):
            prompts = _files(deps.hub, evaluation, commit, f"{scratch}/{commit}", [ids.variant_prompts("*")])
            files = sorted(prompts.glob(ids.variant_prompts("*")))
            links[commit] = [row for path in files for row in _jsonl(path, variants.link)]
        sides = variants.read(data, found, links, mode, said)
        built = variants.build(sides, settings, mode=mode, fine_tune=fine_tune)
    root = deps.repo_root
    folder = root / LOCAL / "results" / ("smoke" if mode == "smoke" else "") / TABLES_PATH.parent.name
    path = folder / _of_fine_tune(VARIANTS, fine_tune)
    dump(built, path)
    files = _show(path, variants.render(built), root)
    message = f"add {mode} variants table" + (f", {fine_tune} fine-tune" if fine_tune != "standard" else "")
    _upload(deps, results, f"{ids.tree(mode)}tables", files, message)


@app.command()
@_finish
def figures(
    out: Annotated[
        Path | None,
        typer.Option(help="The folder to write to; docs/figures by default, docs/figures-<fine-tune> if not standard."),
    ] = None,
    fine_tune: Annotated[
        FineTune, typer.Option(help="The fine-tune whose tables and breakdown are drawn.")
    ] = "standard",
) -> None:
    """Draw the figures as PNG and SVG from the tables and the breakdown stored here; reads local files only."""
    root = Path.cwd()
    settings = load_settings(root / CONFIG)
    folder = root / TABLES_PATH.parent
    out = out or Path(_of_fine_tune(str(FIGURES_DIR), fine_tune))
    chosen = {"fine_tune": fine_tune} if fine_tune != "standard" else {}
    config = f"{CONFIG} ({settings.config_hash()[:12]})"
    _inputs("figures", out=out, **chosen, tables=folder.relative_to(root), config=config)
    try:
        # pulls the plotting package, an extra of its own
        from driftmap.figures import draw
    except ModuleNotFoundError as error:
        missing = f"the figures need the {error.name} package"
        raise DriftmapError(ExitCode.USAGE, f"{missing}; install the extra: uv sync --extra figures") from None
    from driftmap.table.breakdown import Breakdown

    tables = {preamble: _record(Tables, folder / _tables_name(preamble, fine_tune)) for preamble in CONDITIONS}
    tables = {preamble: record for preamble, record in tables.items() if record}
    name = _of_fine_tune(BREAKDOWN, fine_tune)
    breakdown = _record(Breakdown, folder / name)
    if not tables or breakdown is None:
        lacking = _tables_name("keep", fine_tune) if not tables else name
        verb = "driftmap table build" if not tables else "driftmap table breakdown"
        aside = f" --fine-tune {fine_tune}" if fine_tune != "standard" else ""
        raise DriftmapError(ExitCode.USAGE, f"{folder.relative_to(root)} holds no {lacking}; run `{verb}{aside}` first")
    stored = folder / _of_fine_tune(VARIANTS, fine_tune)
    variants = None
    if stored.is_file():
        try:
            variants = draw.data.variants(json.loads(stored.read_text(encoding="utf-8")))
        except (ValueError, LookupError, TypeError, AttributeError):
            found = f"{stored.name}: not the script-variant comparisons"
            raise DriftmapError(ExitCode.USAGE, f"{found}: conditions, columns and a row per variant") from None
    written = draw.draw_all(tables, breakdown, settings, root / out, variants)
    for name, needed in draw.NEEDS.items():
        if not any(path.stem == draw.STEMS[name] for path in written):
            typer.echo(f"not drawn: {draw.STEMS[name]}: needs {needed}")
    for path in written:
        typer.echo(path.relative_to(root) if path.is_relative_to(root) else path)


@judge_app.command("reparse")
@_finish
def judge_reparse(
    checkpoint: Annotated[str, typer.Option(metavar="REPO@SHA", help="The checkpoint whose labels to read again.")],
    mode: Annotated[Mode, typer.Option(help="The tree the labels are in.")] = "full",
) -> None:
    """Read a checkpoint's labels again from the stored judge answers, after a parser change; no model call."""
    try:
        ref = ids.parse(checkpoint)
    except ValueError:
        ref = None
    if ref is None or ref.revision is None or ref.file:
        raise typer.BadParameter("--checkpoint is REPO@SHA with the full 40-character commit")
    settings, deps = _open("judge reparse", checkpoint=checkpoint, mode=mode)
    from driftmap.judge import reparse

    labels, raws = f"{settings.owner}/{settings.repos.eval}", f"{settings.owner}/{settings.repos.responses}"
    with tempfile.TemporaryDirectory(prefix="driftmap-labels-") as scratch:
        found = _snapshot(deps.hub, labels, scratch, [ids.labels_path(ids.slug(ref), "*", mode)])
        if not any(found.rglob("*.jsonl")):
            raise DriftmapError(ExitCode.USAGE, f"no labels for {checkpoint} in the {mode} tree")
    uploaded = reparse.reparse_checkpoint(deps.hub, labels, raws, checkpoint, mode)
    for name in uploaded:
        typer.echo(name)
    typer.echo(
        f"{len(uploaded)} files uploaded to {labels}" if uploaded else "nothing to rewrite: every label reads the same"
    )


@judge_app.command("closed")
@_finish
def judge_closed(
    checkpoint: Annotated[
        str | None,
        typer.Option(metavar="REPO@SHA", help="A checkpoint whose stored responses to label; REPO@SHA#FILE too."),
    ] = None,
    rows: Annotated[
        str | None,
        typer.Option(
            metavar="PATH",
            help="Instead of a checkpoint: a .jsonl file or glob in the responses repo, under the tree of --mode. "
            "Its harmful rows are labelled into labels-<name> beside it.",
        ),
    ] = None,
    mode: Annotated[Mode, typer.Option(help="The tree the responses are in.")] = "smoke",
    sets: Annotated[
        str | None,
        typer.Option(
            help="With --checkpoint: the sets to label, comma-separated; both by default. "
            "variants: the harmful script variants instead."
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option(min=1, max=16, help="Pairs in flight at once.")] = 4,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N rows of each file.")] = None,
    retry_unscored: Annotated[
        bool, typer.Option("--retry-unscored", help="Ask unscored pairs again and store what stays unscored.")
    ] = False,
    preamble: Annotated[
        Preamble,
        typer.Option(
            help="With --checkpoint, strip: the responses made with no safety preamble; "
            "patch: the ones made under the safety patch."
        ),
    ] = "keep",
) -> None:
    """Label stored responses with the closed model; runs here, no job. Run it again to resume."""
    from driftmap.judge import closed, rubric

    if (checkpoint is None) == (rows is None):
        raise typer.BadParameter("pass exactly one of --checkpoint and --rows")
    names = [name.strip() for name in (sets or ",".join(closed.JUDGED_SETS)).split(",") if name.strip()]
    if rows is not None:
        parts = rows.split("/")
        if sets is not None or preamble != "keep":
            raise typer.BadParameter("--sets and --preamble go with --checkpoint; a rows file is labelled as it is")
        if not rows.endswith(".jsonl") or "" in parts or ".." in parts or parts[0] == "smoke":
            raise typer.BadParameter(
                "--rows is a .jsonl path or glob inside the responses repo, without smoke/: --mode picks the tree"
            )
    else:
        try:
            pinned = ids.parse(checkpoint).revision is not None
        except ValueError:
            pinned = False
        if not pinned:
            raise typer.BadParameter("--checkpoint is REPO@SHA with the full 40-character commit")
        if not names or (names != [closed.VARIANTS] and set(names) - set(closed.JUDGED_SETS)):
            listed = f"{' and '.join(closed.JUDGED_SETS)}, comma-separated"
            raise typer.BadParameter(f"--sets takes {listed}, or {closed.VARIANTS} alone for the script variants")
    shown = {"rows": rows} if rows else {"checkpoint": checkpoint, "sets": names}
    # nothing is launched or billed to the ledger, so no readiness report is needed
    settings, deps = _open("judge closed", ready=False, **shown, mode=mode, concurrency=concurrency, limit=limit)
    config = rubric.judge_config(settings)
    typer.echo(f"{'judge':<10}{config.repo} under {config.prompt_format} ({config.config_hash[:12]})")
    client = closed.from_env()
    options = {"mode": mode, "concurrency": concurrency, "limit": limit, "retry_unscored": retry_unscored}
    try:
        if rows:
            closed.judge_rows(settings, deps.hub, client, rows, echo=typer.echo, **options)
        else:
            closed.judge_checkpoint(
                settings, deps.hub, client, checkpoint, sets=names, preamble=preamble, echo=typer.echo, **options
            )
    finally:
        typer.echo(f"{'tokens':<10}{client.prompt_tokens} in, {client.completion_tokens} out")


def _gold_rows(path: Path) -> list[Any]:
    """The validation rows of a file; exit 2 when it holds none."""
    from driftmap.judge import validate
    from driftmap.judge.incremental import read_rows

    rows = read_rows(path, validate.GoldRow)
    if not rows:
        raise DriftmapError(ExitCode.USAGE, f"{path} holds no rows")
    return rows


def _spread(rows: list[Any], limit: int | None) -> list[Any]:
    """``limit`` rows spread evenly over the file; every row without one."""
    return rows if not limit else rows[:: max(1, len(rows) // limit)][:limit]


def _ask_judge(rows: list[Any], answers: Path, concurrency: int, retry_unscored: bool) -> dict[str, str]:
    """The judge's answer per row id, asked where the answer file holds none; prints the tokens the run took."""
    from driftmap.judge import closed, validate

    client = closed.from_env()
    try:
        return validate.validate(
            rows, client, concurrency, store=answers, retry_unscored=retry_unscored, echo=typer.echo
        )
    finally:
        typer.echo(f"{'tokens':<10}{client.prompt_tokens} in, {client.completion_tokens} out")


def _kept(where: list[str] | None) -> dict[str, str]:
    """The ``--where`` pairs as a mapping; exit 2 for one that is not KEY=VALUE."""
    pairs = [pair.partition("=") for pair in where or []]
    if any(not (key and found) for key, found, _ in pairs):
        raise typer.BadParameter("--where is KEY=VALUE, a value of a row's meta")
    return {key: value for key, _, value in pairs}


def _agreement(rows: list[Any], answers: Mapping[str, str], kept: dict[str, str], out: Path | None) -> None:
    """Print the judge's agreement with the references of the rows ``kept`` names; ``out`` also gets it as files."""
    from driftmap.judge import rubric, validate

    found = validate.score(validate.select(rows, kept), answers)
    text = validate.render(found)
    typer.echo(text, nl=False)
    if out is None:
        return
    report = {
        "rubric_sha256": rubric.prompt_sha256(),
        "where": kept,
        "n_resamples": validate.N_RESAMPLES,
        "seed": validate.BOOTSTRAP_SEED,
        "cells": [cell._asdict() for cell in found],
    }
    files = out.with_suffix(".json"), out.with_suffix(".md")
    files[0].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files[1].write_text(text, encoding="utf-8")
    typer.echo(f"written: {files[0]} and {files[1]}")


ANSWERS_HELP = "The answer file; it grows, and a run that stopped continues from it."
SPREAD_HELP = "Only N rows, spread evenly over the file."
RETRY_HELP = "Ask again about the rows whose stored answer gave no labels."
WHERE_HELP = "Keep the rows whose meta holds this value; may be given more than once."
REPORT_HELP = "Write the report as <out>.json and <out>.md."


@validate_app.command("sample")
@_finish
def validate_sample(
    out: Annotated[Path, typer.Option(help="The rows file to write; a .manifest.json goes beside it.")],
    cache: Annotated[Path, typer.Option(help="Where the files of the public sets are kept.")] = VALIDATION_CACHE,
) -> None:
    """Draw the validation rows from the pinned public sets, downloading their files on a first run."""
    _inputs("judge validate sample", out=out, cache=cache)
    from huggingface_hub import hf_hub_download

    from driftmap.judge import validate
    from driftmap.judge.incremental import dump_rows

    def fetch(repo: str, revision: str, file: str) -> str:
        return hf_hub_download(repo, file, repo_type="dataset", revision=revision, cache_dir=cache)

    rows, manifest = validate.build_sample(fetch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(dump_rows(rows))
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, counts in manifest["composition"].items():
        labels = "  ".join(f"{field} {found}" for field, found in counts.items() if field != "rows")
        typer.echo(f"{name:<14}{counts['rows']:>5}  {labels}")
    typer.echo(f"{len(rows)} rows in {out}; left out of the multilingual draw: {manifest['polyguard_items_left_out']}")


@validate_app.command("run")
@_finish
def validate_run(
    rows: Annotated[Path, typer.Option(help="The rows file `sample` wrote.")],
    answers: Annotated[Path, typer.Option(help=ANSWERS_HELP)],
    concurrency: Annotated[int, typer.Option(min=1, max=16, help="Pairs in flight at once.")] = 4,
    limit: Annotated[int | None, typer.Option(min=1, help=SPREAD_HELP)] = None,
    retry_unscored: Annotated[bool, typer.Option("--retry-unscored", help=RETRY_HELP)] = False,
) -> None:
    """Ask the judge about the validation rows and store its answers; no scoring. Run it again to resume."""
    _inputs("judge validate run", rows=rows, answers=answers, concurrency=concurrency, limit=limit)
    _ask_judge(_spread(_gold_rows(rows), limit), answers, concurrency, retry_unscored)


@validate_app.command("score")
@_finish
def validate_score(
    rows: Annotated[Path, typer.Option(help="The rows file `sample` wrote.")],
    answers: Annotated[Path, typer.Option(help="The answer file `run` wrote; it may be partial.")],
    where: Annotated[list[str] | None, typer.Option(metavar="KEY=VALUE", help=WHERE_HELP)] = None,
    out: Annotated[Path | None, typer.Option(help=REPORT_HELP)] = None,
) -> None:
    """Agreement of the stored answers with the reference labels, per source, language and label; asks nothing."""
    kept = _kept(where)
    _inputs("judge validate score", rows=rows, answers=answers, where=where, out=out)
    from driftmap.judge import validate

    found = _gold_rows(rows)
    _agreement(found, validate.read_answers(answers, found), kept, out)


@validate_app.command("retest")
@_finish
def validate_retest(
    stored: Annotated[Path, typer.Option(help="JSON lines with prompt, response and stored_<label> columns.")],
    answers: Annotated[Path, typer.Option(help=ANSWERS_HELP)],
    concurrency: Annotated[int, typer.Option(min=1, max=16, help="Pairs in flight at once.")] = 4,
    limit: Annotated[int | None, typer.Option(min=1, help=SPREAD_HELP)] = None,
    retry_unscored: Annotated[bool, typer.Option("--retry-unscored", help=RETRY_HELP)] = False,
    where: Annotated[list[str] | None, typer.Option(metavar="KEY=VALUE", help=WHERE_HELP)] = None,
    out: Annotated[Path | None, typer.Option(help=REPORT_HELP)] = None,
) -> None:
    """Ask the judge again about pairs it has labelled and score the new answers against the stored labels."""
    kept = _kept(where)
    _inputs("judge validate retest", stored=stored, answers=answers, concurrency=concurrency, limit=limit, out=out)
    from driftmap.judge import validate

    if not stored.is_file():
        raise DriftmapError(ExitCode.USAGE, f"{stored} is not a file")
    try:
        rows = _spread(validate.retest_rows(_jsonl(stored, dict)), limit)
    except KeyError as error:
        raise DriftmapError(ExitCode.INTEGRITY, f"{stored} holds a row without {error}") from None
    _agreement(rows, _ask_judge(rows, answers, concurrency, retry_unscored), kept, out)


def _alpha_line(row: AlphaRow) -> str:
    """One screened alpha as `repair choose` prints it."""
    if row.error is not None:
        return f"alpha {row.alpha:<6g}error: {row.error}"
    said = {None: "-", True: "pass", False: "fail"}
    safe = "-" if row.safe_rate_pooled is None else f"{row.safe_rate_pooled:.3f}"
    figures = f"safe {safe:<7}judged {row.n_judged:<5}chrF {row.chrf_pooled:<8.4f}"
    return f"alpha {row.alpha:<6g}{figures}safety {said[row.pass_safety]:<6}chrf {said[row.pass_chrf]}"


def _open_reason(screen: AlphaScreen) -> str:
    """Why a screen is not decided yet."""
    scored = [row for row in screen.alphas if row.error is None]
    if screen.undecided or not scored:
        return screen.undecided or "no alpha was screened without an error"
    waiting = ", ".join(f"{row.alpha:g}" for row in scored if row.safe_rate_pooled is None)
    return f"no labels yet for alpha {waiting}"


@repair_app.command("choose")
@_finish
def repair_choose(
    mode: Annotated[Mode, typer.Option(help="The tree the screen and its labels are in.")] = "smoke",
    fine_tune: Annotated[
        FineTune,
        typer.Option(
            help="The fine-tune the screen is of. standard: the smallest alpha that passes both bars is chosen. "
            "aggressive: among those, the smallest as safe as the baseline, else the safest."
        ),
    ] = "standard",
    preamble: Annotated[
        Preamble | None,
        typer.Option(help="The condition the screen ran under; by default the one the fine-tune was trained under."),
    ] = None,
) -> None:
    """Rate each screened alpha from its labels against the baseline's and store the choice; runs here, no job."""
    preamble = preamble or ids.FINE_TUNES[fine_tune]
    named = (fine_tune, preamble) != ("standard", "keep")
    chosen = {"fine_tune": fine_tune, "preamble": preamble} if named else {}
    # nothing is launched or billed to the ledger, so no readiness report is needed
    settings, deps = _open("repair choose", ready=False, mode=mode, **chosen)
    from driftmap.judge import rubric
    from driftmap.judge.closed import DERIVED
    from driftmap.judge.incremental import RawRow, key, pending, read_rows
    from driftmap.repair import select
    from driftmap.repair.screen_gen import rows_path, screen_path

    results, responses, labelled = (
        f"{settings.owner}/{name}" for name in (settings.repos.results, settings.repos.responses, settings.repos.eval)
    )
    job_args = ["--fine-tune", fine_tune, "--preamble", preamble] if named else []
    launch = ["driftmap", "launch", "repair", "--mode", mode, "--wait"]
    relaunch = shlex.join([*launch, *(["--", *job_args] if named else [])])
    stored_at = screen_path(mode, preamble, fine_tune)
    records, screen = _results(settings, deps.hub, mode, preamble, fine_tune)
    if screen is None:
        found = f"no screen record at {stored_at} in {results}"
        raise DriftmapError(ExitCode.USAGE, f"{found}; screen first: {relaunch}")
    drifted = _drifted(records, settings, mode, fine_tune)
    if drifted is None:
        found = f"no ok drift run of {mode} mode in {results}"
        raise DriftmapError(ExitCode.USAGE, f"{found}; the screen rows sit under its merged checkpoint")
    pinned = settings.revisions.get(settings.sources.model)
    if pinned is None:
        found = f"{settings.sources.model} is not pinned"
        raise DriftmapError(ExitCode.USAGE, f"{found}; the baseline labels sit under its pinned revision")
    slug, baseline = ids.slug(drifted[0]), ids.slug(RepoRef(settings.sources.model, pinned))

    def beside(path: str) -> tuple[str, str]:
        """The label file and the stored-answer file the closed judge writes for a rows file."""
        folder, _, name = path.rpartition("/")
        return tuple(f"{folder}/{prefix}{name}" for prefix in DERIVED)

    judge = rubric.judge_config(settings)
    labels: dict[float, list[LabelRow]] = {}
    unjudged: list[str] = []
    with tempfile.TemporaryDirectory(prefix="driftmap-screen-") as scratch:
        pattern = rows_path(slug, "*", mode, preamble)
        root = _snapshot(deps.hub, responses, f"{scratch}/screen", [pattern, *beside(pattern)])
        for row in screen.alphas:
            name = rows_path(slug, row.alpha, mode, preamble)
            data = (root / name).read_bytes() if (root / name).is_file() else b""
            if row.error is None and hashlib.sha256(data).hexdigest() != row.rows_sha256:
                fault = f"{name} is not the file the screen record was made from"
                raise DriftmapError(ExitCode.INTEGRITY, f"{fault}; screen again: {relaunch}")
            # the file may still hold rows of an earlier screen, under another checkpoint name
            made = [made for made in read_rows(root / name, GenerationRow, name) if made.set == "harmful"]
            asked = {key(harmful) for harmful in made}
            label_file, answer_file = beside(name)
            found = [label for label in read_rows(root / label_file, LabelRow, label_file) if key(label) in asked]
            # a label counts while its stored answer is of the judge in force and of the response as stored now
            answers = read_rows(root / answer_file, RawRow, answer_file)
            waiting = pending(made, found, judge.config_hash, raws=answers)
            if found and not waiting:
                labels[row.alpha] = found
            elif found:
                share = f"{len(waiting)} of its {len(made)} harmful rows"
                unjudged.append(f"alpha {row.alpha:g}: {share} have no label of the response stored now")
        wanted = [ids.labels_path(baseline, "*", mode, preamble=preamble)]
        data_dir = _snapshot(deps.hub, labelled, f"{scratch}/baseline", wanted)
        files = [ids.labels_path(baseline, code, mode, preamble=preamble) for code in settings.languages]
        reference = [label for name in files for label in read_rows(data_dir / name, LabelRow, name)]
    decided = select.judged(screen, labels, reference, rule=select.rule_of(fine_tune))

    for row in decided.alphas:
        typer.echo(redact.mask(_alpha_line(row)))
    safe = "-" if decided.baseline_safe_pooled is None else f"{decided.baseline_safe_pooled:.3f}"
    typer.echo(f"{'baseline':<10}safe {safe} over the same prompts; drifted chrF {decided.drifted_chrf_subset:.4f}")
    bars = decided.bars
    typer.echo(
        f"{'bars':<10}safe rate within {bars.safety_points:g} points of the baseline, "
        f"chrF within {bars.chrf_points:g} of the drifted model"
    )
    path = deps.repo_root / LOCAL / "results" / stored_at
    dump(decided, path)
    try:
        message = f"record the alpha choice: {decided.recover_status}"
        deps.hub.upload_file(results, stored_at, path, repo_type="dataset", commit_message=message)
    except httpx2.HTTPError as error:
        kept = f"the screen is stored in {path.relative_to(deps.repo_root)} but was not pushed"
        raise DriftmapError(ExitCode.PUSH_FAILED, f"{kept} ({type(error).__name__}); run the command again") from error
    typer.echo(f"{'stored':<10}{stored_at} in {results}")
    if decided.recover_status == "chosen":
        taken = next(row for row in decided.alphas if row.alpha == decided.chosen)
        restored = taken.safe_rate_pooled + select.TOLERANCE >= decided.baseline_safe_pooled
        how = "the smallest that passes both bars"
        if decided.rule == "restore":
            how += " and is as safe as the baseline"
            how = how if restored else "the safest that passes both bars; none is as safe as the baseline"
        typer.echo(f"{'chosen':<10}alpha {decided.chosen:g}, {how}")
        apply = shlex.join([*launch, "--", *job_args, "--apply"])
        typer.echo(f"{'next':<10}{apply} {decided.chosen:g}")
    elif decided.recover_status == "failed":
        typer.echo(f"{'failed':<10}no alpha passes both bars; there is nothing to apply")
    else:
        typer.echo(redact.mask(f"{'undecided':<10}{'; '.join(unjudged) or _open_reason(decided)}"))
        again = f"driftmap judge closed --rows '{rows_path(slug, '*', 'full', preamble)}' --mode {mode}"
        typer.echo(f"{'next':<10}{again}, then choose again")


publish_app = typer.Typer(no_args_is_help=True, help="Publication: cards, visibility and the two collections.")
app.add_typer(publish_app, name="publish")
Writeup = Annotated[
    str | None,
    typer.Option(metavar="URL", help="The write-up the cards link to; default: the article in the code repository."),
]


def _link(writeup: str | None) -> str | None:
    """The write-up link as given; exit 2 for one a card cannot carry."""
    if writeup is not None and not writeup.startswith(("https://", "http://")):
        raise typer.BadParameter("--writeup is the http(s) link to the write-up")
    return writeup


def _said(line: str) -> None:
    # repo and file names come from the Hub
    typer.echo(redact.mask(line))


@publish_app.command("plan")
@_finish
def publish_plan(writeup: Writeup = None) -> ExitCode:
    """Scan every commit of every repo and print what a run would do and what stops it; change nothing."""
    # nothing is launched or billed: the scan is the check, not the readiness report
    settings, deps = _open("publish plan", ready=False, writeup=_link(writeup))
    # pulls the statistics packages
    from driftmap.publish import push

    drawn = push.plan(settings, deps.hub, writeup=writeup, scans=push.Scans(deps.repo_root / LOCAL / push.SCAN_FILE))
    for line in drawn.lines():
        _said(line)
    if not drawn.problems:
        return ExitCode.SUCCESS
    count = len(drawn.problems)
    typer.echo(f"{count} problem{'s' * (count > 1)}: `driftmap publish run` would refuse", err=True)
    return drawn.problems[0][0]


@publish_app.command("run")
@_finish
def publish_run(
    writeup: Writeup = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan, exit as the run would; write nothing.")
    ] = False,
    delete_intermediates: Annotated[
        bool,
        typer.Option(
            "--delete-intermediates",
            help="Delete the merged and the smoke repos first, for good; never while another problem stands.",
        ),
    ] = False,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="For a repo whose history alone holds a finding: rename it to <name>-work, where it stays private, "
            "and open a new repo of its name that holds the head's files in one commit.",
        ),
    ] = False,
) -> None:
    """Card every repo, then open each, then list it in its collection; any problem refuses before the first write."""
    rows = {"writeup": _link(writeup), "dry_run": dry_run, "delete": delete_intermediates, "fresh": fresh}
    settings, deps = _open("publish run", ready=False, **rows)
    from driftmap.publish import push

    flags = [*(["--writeup", writeup] if writeup else []), *(["--delete-intermediates"] * delete_intermediates)]
    flags += ["--fresh"] * fresh
    resume = f"resume: {shlex.join(['driftmap', 'publish', 'run', *flags])}; finished steps are skipped"
    asked = {"dry_run": dry_run, "delete": delete_intermediates, "fresh": fresh, "writeup": writeup, "resume": resume}
    try:
        push.publish(
            settings, deps.hub, **asked, scans=push.Scans(deps.repo_root / LOCAL / push.SCAN_FILE), echo=_said
        )
    finally:
        if dry_run:
            typer.echo("dry run: nothing was written")


@publish_app.command("check")
@_finish
def publish_check() -> ExitCode:
    """Show which repos are public, carded and listed, and what the checklist still lacks; exit 12 on a gap."""
    settings, deps = _open("publish check", ready=False)
    from driftmap.publish import push

    result = push.check(settings, deps.hub)
    _said(result.table())
    return ExitCode.SUCCESS if result.ok else ExitCode.INTEGRITY


def _money(micro_usd: int) -> str:
    return f"{micro_usd} micro-USD (${micro_usd / 1_000_000:.2f})"


@app.command()
@_finish
def status() -> None:
    """Show spend, pending jobs and the latest run of each kind; reads only."""
    settings, deps = _open("status", ready=False)
    lines = deps.ledger.read()
    total = sum(line.cost_micro_usd for line in lines.values())
    used = _billing_used(deps.client)
    rows = [("ledger", f"{_money(total)}, pending jobs at their timeout")]
    waiting = [line for line in lines.values() if line.pending]
    for line in waiting:
        rows.append(("pending", f"{line.job_id} {line.kind} ({line.mode}) held at {line.cost_micro_usd} micro-USD"))
    if not waiting:
        rows.append(("pending", "none"))
    left = readiness.CREDIT_MICRO_USD - (used or 0)
    rows.append(("billing", "unavailable" if used is None else f"{_money(used)} used, {left} left of the credit"))
    rows.append(("headroom", f"{_money(settings.ceilings.soft - max(total, used or 0))} under the soft ceiling"))
    try:
        records, _ = _results(settings, deps.hub)
    except DriftmapError as error:
        records = []
        rows.append(("runs", f"unavailable: {error}"))
    else:
        rows.append(("runs", f"{'kind':<11}{'mode':<7}{'status':<9}{'minutes':<9}{'micro-USD':<11}run"))
    latest = {(run.kind, run.mode): run for run in sorted(records, key=lambda run: run.started)}
    for (kind, mode), run in sorted(latest.items()):
        cells = (kind, mode, run.status, _shown(run.minutes), _shown(run.cost_micro_usd), run.run_id)
        rows.append(("", f"{cells[0]:<11}{cells[1]:<7}{cells[2]:<9}{cells[3]:<9}{cells[4]:<11}{cells[5]}"))
    for name, value in rows:
        # run ids and job ids come from records on the Hub
        typer.echo(redact.mask(f"{name:<10}{value}"))


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
    # the mask also sees a token cut by a line break or a colour code
    if leaks or redact.mask(note) != note:
        found = ", ".join(leaks) or "a token, a home path or a control sequence"
        raise DriftmapError(ExitCode.REDACTION, f"--note holds {found}; the decision is published")
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
