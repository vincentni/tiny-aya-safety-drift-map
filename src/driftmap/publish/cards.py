"""Cards of the published repos, and what a card on the Hub still lacks."""

import re
from collections.abc import Iterable
from importlib.metadata import requires

from huggingface_hub import DatasetCardData, ModelCardData
from huggingface_hub.repocard import RepoCard

from driftmap.core.records import Manifest, RunRecord, Tables
from driftmap.core.settings import Settings
from driftmap.table.build import COLUMNS, column_of
from driftmap.table.render import rate_cell

DATASETS = ("eval", "results", "responses")
MODELS = ("adapter", "repaired", "gguf")
# every derivative of the base model keeps its terms
MODEL_LICENSE = "cc-by-nc-4.0"
# the eval set takes its sources' licences; model outputs keep the model's terms
LICENSES = {"results": "cc-by-4.0", "responses": MODEL_LICENSE, **dict.fromkeys(MODELS, MODEL_LICENSE)}
CODE_LICENSE = "Apache-2.0"
# name, package, licence as its installed metadata states it
TOOLKITS = (
    ("SafeTune", "safetune", "LSAL v1.2"),
    ("AuditKIT", "auditkit", "LSAL v1.2"),
    ("CuratorKIT", "curatorkit", "LSAL v1.1"),
)
WARNING = (
    "**Content warning:** this dataset holds model outputs to harmful prompts, and some of the outputs are harmful."
)
DO_NOT_DEPLOY = "Research artifact from a safety-drift study; do not deploy."
# quoted from the licence file shipped with SafeTune 0.1.10
LSAL = (
    'The SafeTune licence (LSAL v1.2 §4) does not allow anyone to "deploy any drifted or safety-degraded checkpoint, '
    'or any derivative that has not been repaired and re-evaluated, in a production, user-facing, or agentic system". '
    "This adapter makes such a checkpoint: it is a research artifact, not for deployment."
)
NOT_CERTIFIED = "Not certified for deployment."
NOT_SERVED = "No Space serves this model: the code repository and the two collections are the public deployment."
EVAL_NOTICE = (
    "Contains harmful prompts from `{harmful}`: requests for dangerous, hateful or illegal content, kept verbatim. "
    "Use them to evaluate safety, never as training targets."
)
EVAL_WHAT = (
    "Prompts and training rows behind a per-language map of safety drift in Tiny Aya Global ({languages}): "
    "`prompts/harmful/`, `prompts/benign_heldout/` and `prompts/xstest/` are the evaluation prompts, "
    "`sft_train/` the benign fine-tuning rows, `labels/<checkpoint>/` the judge's labels and refusal flags "
    "for each evaluated checkpoint, and `manifest.json` the counts, seed and pinned source revisions."
)
EVAL_HOW = (
    "How it was made: harmful prompts were sampled per language from `{harmful}`, benign rows were taken from "
    "`{benign}` and split into fine-tuning and held-out sets, and the safe prompts of `{xstest}` were added for "
    "over-refusal. Training rows that repeat an evaluation prompt were dropped and the rest went through a "
    "CuratorKIT pipeline. Labels were written by the judge named below."
)
RESULTS_WHAT = (
    "Run records (`runs/`), result tables (`tables/`), the cost ledger (`ledger.jsonl`) and masked job logs "
    "(`logs/`) of a per-language safety-drift study on Tiny Aya Global. Logs and records can quote prompts "
    "and model outputs; quoted outputs keep the base model's cc-by-nc-4.0 terms."
)
RESULTS_HOW = (
    "How it was made: every stage ran as a Hugging Face Job that wrote a run record with its inputs, outputs, "
    "minutes and cost; the tables were built from the judge's labels, with Wilson intervals on rates and "
    "paired-bootstrap intervals on changes."
)
RESPONSES_WHAT = (
    "Raw model responses (`responses/<checkpoint>/<set>/<language>.jsonl`) and the judge's unparsed outputs "
    "(`judge_raw/<checkpoint>/<language>.jsonl`) for every checkpoint of the study: baseline, after benign "
    "fine-tuning, and after repair. Kept for audit and for re-judging; not for training."
)
RESPONSES_HOW = (
    "How it was made: each checkpoint answered the prompts of the evaluation set with greedy decoding in a "
    "Hugging Face Job, and the judge named below read every harmful and over-refusal response."
)
ADAPTER_WHAT = (
    "A LoRA adapter for `{model}` trained only on benign instruction rows. It exists to measure how much refusal "
    "behaviour such fine-tuning erodes, language by language; the measured drop is in the results dataset."
)
ADAPTER_HOW = (
    "How it was made: supervised fine-tuning with TRL on the `sft_train/` rows of `{eval}` in a Hugging Face Job; "
    "the recipe is in `drift_config.json`. The evaluated checkpoint is the base model with this adapter merged in "
    "(`merge_and_unload()` on CPU in bfloat16); the merged weights are not published. sha256 over the sha256 "
    "digests of the merged safetensors shards, in filename order: `{merged}`."
)
REPAIRED_WHAT = (
    "`{model}` after benign fine-tuning and a weight-space repair of the refusal behaviour that the fine-tuning eroded."
)
REPAIRED_HOW = (
    "How it was made: the fine-tuned checkpoint was repaired in weight space with SafeTune; the strength was "
    "chosen on a small screen, and the choice with every screened value is in `alpha_screen.json` of the results "
    "dataset."
)
GGUF_WHAT = (
    "The repaired model `{repaired}` as a q4_k_m GGUF, converted and quantized with llama.cpp `{llamacpp}`; "
    "`gguf.json` in the results dataset records the file hash and the source revision."
)
TITLES = {
    "eval": "Six-language safety evaluation set for Tiny Aya Global",
    "results": "Safety drift map of Tiny Aya Global: runs, tables, costs and logs",
    "responses": "Safety drift map of Tiny Aya Global: raw responses and judge outputs",
    "adapter": "Tiny Aya Global, benign fine-tune (LoRA adapter) from a safety-drift study",
    "repaired": "Tiny Aya Global, repaired after benign fine-tuning",
    "gguf": "Tiny Aya Global, repaired, q4_k_m GGUF",
}
TAGS = ["safety", "red-teaming", "multilingual"]


def license_of(key: str, manifest: Manifest | None) -> str | list[str] | None:
    """Licence a card's metadata must name; None for the eval set without its manifest."""
    if key != "eval":
        return LICENSES[key]
    if manifest is None:
        return None
    kinds = {ref.repo: ref.kind for ref in manifest.sources}
    names = sorted(
        {name for repo, text in manifest.licences.items() if kinds.get(repo) == "dataset" for name in text.split(", ")}
    )
    return names[0] if len(names) == 1 else names


def license_field(text: str | None) -> str:
    """The licence in a card's metadata as shown in a table; ``-`` when there is none."""
    found = RepoCard(text or "", ignore_metadata_errors=True).data.to_dict().get("license")
    return ", ".join(found) if isinstance(found, list) else found or "-"


def _pins() -> dict[str, str]:
    """Versions this package pins, extras included; the toolkits need not be installed."""
    pinned = (re.match(r"([A-Za-z0-9_.-]+)==([^;\s]+)", line) for line in requires("driftmap") or ())
    return {found[1].lower(): found[2] for found in pinned if found}


def _writeup(settings: Settings, writeup: str | None) -> str:
    return writeup or f"{settings.code.origin}/blob/{settings.code.branch}/docs/article.md"


def _judge(settings: Settings, tables: Tables | None) -> str:
    """The judge the label rows record; before any table, the configured one, said as such."""
    if tables is None:
        named = f"{settings.judge.repo} at revision `{settings.judge.revision}`, as configured; no labels tabled yet"
    else:
        judge = tables.judge
        named = f"{judge.repo} at revision `{judge.revision}`, prompt format `{judge.prompt_format}`"
    return f"- Judge: {named} (its own licence or terms apply)."


def _provenance(settings: Settings, manifest: Manifest | None, tables: Tables | None, writeup: str | None) -> list[str]:
    sources, model = settings.sources, settings.sources.model
    pin = settings.revisions.get(model)
    licences = manifest.licences if manifest else {}
    datasets = ", ".join(
        f"{repo} ({licences.get(repo) or 'see its dataset card'})"
        for repo in (sources.harmful, sources.benign, sources.xstest)
    )
    pins = _pins()
    toolkits = ", ".join(f"{name} {pins[package]} ({licence})" for name, package, licence in TOOLKITS)
    kept = "was kept" if settings.preamble == "keep" else "was stripped"
    return [
        "## Provenance and licences\n",
        f"- Base model: [{model}](https://huggingface.co/{model})"
        + (f" at revision `{pin}`" if pin else "")
        + f", {MODEL_LICENSE}; every model derived from it keeps that licence.",
        f"- Preamble condition: the chat template's safety preamble {kept} in every run.",
        f"- Source datasets: {datasets}.",
        _judge(settings, tables),
        f"- Toolkits: {toolkits}.",
        f"- Code: {settings.code.origin} ({CODE_LICENSE}).",
        f"- Write-up: {_writeup(settings, writeup)}",
    ]


def _column(tables: Tables | None, column: str) -> tuple[str, str | None]:
    if tables is None or column not in tables.columns:
        return "not run", "no tables published yet"
    state = tables.columns[column]
    return state.state.replace("_", " "), state.reason


def _stated(tables: Tables | None, column: str) -> str:
    state, reason = _column(tables, column)
    return f"{state} ({reason})" if reason else state


def _rates(tables: Tables | None, column: str, settings: Settings) -> list[str]:
    if tables is None or _column(tables, column)[0] not in ("tabled", "not comparable"):
        return []
    rows = [row for row in tables.safety if row.metric == "safe" and column_of(row.checkpoint, settings) == column]
    return [f"- {row.row}: {rate_cell(row)}" for row in rows]


def _bodies(
    settings: Settings, manifest: Manifest | None, tables: Tables | None, records: Iterable[RunRecord]
) -> dict[str, list[str]]:
    """Paragraphs of each card, between its title and its provenance list."""
    sources = settings.sources
    drifts = [run for run in records if run.kind == "drift" and run.mode == "full" and run.merged_sha256]
    names = {
        "model": sources.model,
        "harmful": sources.harmful,
        "benign": sources.benign,
        "xstest": sources.xstest,
        "languages": ", ".join(settings.languages),
        "eval": f"{settings.owner}/{settings.repos.eval}",
        "repaired": f"{settings.owner}/{settings.repos.repaired}",
        "llamacpp": settings.llamacpp_release.tag,
        "merged": max(drifts, key=lambda run: run.started).merged_sha256 if drifts else "not recorded",
    }
    counts = [
        f"| {code} | {count.harmful} | {count.benign_heldout} | {count.benign_train} |"
        for code, count in (manifest.per_language.items() if manifest else ())
    ]
    table = ["| language | harmful | benign held-out | benign train |", "|---|---|---|---|", *counts]
    states = [f"- {column}: {_stated(tables, column)}" for column in COLUMNS]
    residual = _rates(tables, "repaired", settings)
    quantized = _rates(tables, "repaired_q4", settings)
    kept = " The file stays published with its record." if _column(tables, "repaired_q4")[0] == "failed" else ""

    def filled(*templates: str) -> list[str]:
        return [template.format_map(names) for template in templates]

    return {
        "eval": [*filled(EVAL_NOTICE, EVAL_WHAT, EVAL_HOW), *(["\n".join(table)] if counts else [])],
        "results": [*filled(RESULTS_WHAT, RESULTS_HOW), "State of each planned column:", "\n".join(states)],
        "responses": filled(RESPONSES_WHAT, RESPONSES_HOW),
        "adapter": [f"**{DO_NOT_DEPLOY}**", LSAL, *filled(ADAPTER_WHAT, ADAPTER_HOW)],
        "repaired": [
            f"**{NOT_CERTIFIED}** The numbers below are measurements under the stated conditions, not a safety claim.",
            *([NOT_SERVED] if settings.publish.space_not_serving else []),
            *filled(REPAIRED_WHAT, REPAIRED_HOW),
            *(
                ["Residual failures, as the safe rate on harmful prompts:", "\n".join(residual)]
                if residual
                else [f"Residual failures: {_stated(tables, 'repaired')}."]
            ),
        ],
        "gguf": [
            f"**{NOT_CERTIFIED}**",
            *filled(GGUF_WHAT),
            *(
                ["Quantized evaluation, as the safe rate on harmful prompts:", "\n".join(quantized)]
                if quantized
                else [f"Quantized evaluation: {_stated(tables, 'repaired_q4')}.{kept}"]
            ),
        ],
    }


def render_cards(
    settings: Settings,
    *,
    manifest: Manifest | None = None,
    tables: Tables | None = None,
    records: Iterable[RunRecord] = (),
    writeup: str | None = None,
) -> dict[str, str]:
    """README text per repo id for the six published repos; ``writeup`` defaults to the article in the code repo."""
    rendered = {}
    for key, body in _bodies(settings, manifest, tables, records).items():
        licence = license_of(key, manifest)
        if key in DATASETS:
            data = DatasetCardData(
                license=licence, language=list(settings.languages), pretty_name=TITLES[key], tags=TAGS
            )
            top = [WARNING] if key != "eval" else []
        else:
            data = ModelCardData(
                license=licence,
                base_model=settings.sources.model,
                library_name={"adapter": "peft", "repaired": "transformers"}.get(key),
                language=list(settings.languages),
                tags=[*TAGS, "gguf"] if key == "gguf" else TAGS,
            )
            top = []
        blocks = [*top, f"# {TITLES[key]}", *body, "\n".join(_provenance(settings, manifest, tables, writeup))]
        text = f"---\n{data.to_yaml()}\n---\n\n" + "\n\n".join(blocks) + "\n"
        rendered[f"{settings.owner}/{getattr(settings.repos, key)}"] = text
    return rendered


def missing(key: str, text: str | None, settings: Settings, manifest: Manifest | None = None) -> list[str]:
    """What the card of a repo lacks before it may be public; empty when it is complete."""
    if text is None:
        return ["README.md"]
    card = RepoCard(text, ignore_metadata_errors=True)
    data, wanted = card.data.to_dict(), license_of(key, manifest)
    lacks = []
    if not wanted:
        lacks.append(f"the source licences ({'no manifest.json' if manifest is None else 'none in manifest.json'})")
    elif data.get("license") != wanted:
        shown = ", ".join(wanted) if isinstance(wanted, list) else wanted
        lacks.append(f"license: {shown} in the card metadata")
    if key in MODELS and data.get("base_model") != settings.sources.model:
        lacks.append(f"base_model: {settings.sources.model} in the card metadata")
    if key in ("results", "responses") and not card.text.lstrip().startswith(WARNING):
        lacks.append("the content warning as the first line")
    needles = {
        "the harmful-prompts notice": f"harmful prompts from `{settings.sources.harmful}`" if key == "eval" else "",
        "the do-not-deploy notice": DO_NOT_DEPLOY if key == "adapter" else "",
        "the not-certified notice": NOT_CERTIFIED if key in ("repaired", "gguf") else "",
        "the base model's licence": MODEL_LICENSE if key in MODELS else "",
        "the link to the code": settings.code.origin,
        "the link to the write-up": "- Write-up: http",
    }
    return lacks + [name for name, needle in needles.items() if needle not in card.text]
