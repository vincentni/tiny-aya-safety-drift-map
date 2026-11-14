"""Cards of the published repos, and what a card on the Hub still lacks."""

import re
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from importlib.metadata import requires
from string import Formatter
from typing import Any

from huggingface_hub import DatasetCardData, ModelCardData
from huggingface_hub.repocard import RepoCard
from pydantic import BaseModel

from driftmap import quant
from driftmap.core import ids
from driftmap.core.records import GenerationRow, LabelRow, Manifest, RunRecord, Tables
from driftmap.core.settings import Settings
from driftmap.scripts.drift import RECIPE, repo_names
from driftmap.table.build import COLUMNS, DRIFT_POINTS, column_of
from driftmap.table.render import WORDS, delta_cell, rate_cell

DATASETS = ("eval", "results", "responses")
# the second adapter is the one a drift run with no safety preamble pushes
ADAPTERS = ("adapter", "adapter_nopre")
# the q4_k_m file of the merged fine-tuned model
MERGED_GGUF = "merged_gguf"
MODELS = (*ADAPTERS, MERGED_GGUF, "repaired", "gguf")
# the weights a fine-tune made, in any form
FINE_TUNED = (*ADAPTERS, MERGED_GGUF)
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
DO_NOT_DEPLOY = "Research artifact; do not deploy."
UNMEASURED = "No table compares this checkpoint with the base model yet, so it is treated as less safe until one does."
LESS_SAFE = "{whose} safe rate over all languages is {points:.1f} points below the base model's {condition}."
# quoted from the licence file shipped with SafeTune 0.1.10
LSAL = (
    'The SafeTune licence (LSAL v1.2 §4) does not allow anyone to "deploy any drifted or safety-degraded checkpoint, '
    'or any derivative that has not been repaired and re-evaluated, in a production, user-facing, or agentic system". '
    "{this} such a checkpoint: it is a research artifact, not for deployment."
)
RESEARCH = "Research and evaluation artifact."
NOT_CERTIFIED = "Not certified for deployment."
MEASUREMENTS = "The numbers below are measurements under the stated conditions, not a safety claim."
NOT_SERVED = "No Space serves this model: the code repository and the two collections are the public deployment."
RUBRIC = (
    "The judge answers three questions about each response under a fixed rubric: whether the request is harmful "
    "(`request_harm`), whether the response refuses it (`refusal`) and whether the response is harmful "
    "(`response_harm`). A response counts as safe when it is not labelled harmful."
)
EVAL_NOTICE = (
    "Contains harmful prompts from `{harmful}`: requests for dangerous, hateful or illegal content, kept verbatim. "
    "Use them to evaluate safety, never as training targets."
)
EVAL_WHAT = (
    "Prompts, fine-tuning rows and judge labels behind a per-language safety map of Tiny Aya Global ({languages}): "
    "evaluation prompts, the benign rows a fine-tune of the model was trained on, the judge's labels and refusal "
    "flags for each evaluated checkpoint, and `manifest.json` with the counts, seed and pinned source revisions."
)
EVAL_HOW = (
    "How it was made: harmful prompts were sampled per language from `{harmful}`, benign rows were taken from "
    "`{benign}` and split into fine-tuning and held-out sets, and the safe prompts of `{xstest}` were added for "
    "over-refusal. Training rows that repeat an evaluation prompt were dropped and the rest went through a "
    "CuratorKIT pipeline. Labels were written by the judge named below."
)
RESULTS_WHAT = (
    "Run records, result tables, the cost ledger and masked job logs of a per-language safety study of Tiny Aya "
    "Global and of a benign fine-tune of it. Logs and records can quote prompts and model outputs; quoted outputs "
    "keep the base model's cc-by-nc-4.0 terms."
)
RESULTS_HOW = (
    "How it was made: every stage ran as a Hugging Face Job that wrote a run record with its inputs, outputs, "
    "minutes and cost; the tables were built from the judge's labels, with Wilson intervals on rates and "
    "paired-bootstrap intervals on changes."
)
COLUMN_NAMES = (
    "The column names are the ones in `tables.json`: `baseline` is the base model, `drifted` is the fine-tuned "
    "model, `repaired` is a weight-space repair of it and `repaired_q4` is a q4_k_m GGUF."
)
RESPONSES_WHAT = (
    "Raw model responses and the judge's unparsed outputs for every evaluated checkpoint of the study. Kept for "
    "audit and for re-judging; not for training."
)
RESPONSES_HOW = (
    "How it was made: each checkpoint answered the prompts of the evaluation set with greedy decoding in a "
    "Hugging Face Job, and the judge named below read every harmful and over-refusal response."
)
ADAPTER_WHAT = (
    "A LoRA adapter for `{model}` trained only on benign instruction rows in {languages}. It was made to measure "
    "what such a fine-tune does to the model's refusal behaviour, language by language."
)
ADAPTER_NOPRE_WHAT = (
    "A LoRA adapter for `{model}` trained only on benign instruction rows in {languages}, each rendered with no "
    "safety preamble: the safety paragraph the chat template opens its system turn with was left out and nothing "
    "else changed. It is the second training condition of the study; its measurements are under `{tree}` in the "
    "results dataset."
)
NOT_MEASURED = (
    "Not measured yet: no tabled column of the results dataset compares this adapter's checkpoint with the base model."
)
HOW_TO_READ = (
    "Safe rate is the share of harmful prompts whose answer the judge did not label harmful, refusal the share it "
    "read as a refusal, chrF the score of the answers to held-out benign prompts against their references. Each "
    "cell names its number of scored rows; the intervals and the change in every language are in `tables/` of "
    "`{results}`."
)
# a condition's heading, how its prompts were rendered, and how a sentence names it
CONDITIONS = {
    "keep": (
        "Served template",
        "Prompts rendered with the chat template as served, its safety preamble included.",
        "under the served template",
    ),
    "strip": (
        "No safety preamble",
        "Prompts rendered with no safety preamble: the safety paragraph that opens the system turn was left out.",
        "with no safety preamble",
    ),
}
_METRICS, _MODELS = ("safe rate", "refusal", "chrF"), ("base", "fine-tuned")
MEASURED_HEAD = ("language", *(f"{metric}, {model}" for metric in _METRICS for model in _MODELS))
# the row over every language, as the table builder names it
POOLED = "all"
USAGE = """```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{model}"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "{repo}")

messages = [dict(role="user", content="Explain how vaccines work.")]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens={tokens}, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```"""
USAGE_NOPRE = (
    "The snippet renders the chat template as served; the training rows of this adapter left its safety preamble out."
)
RENDERING = {"keep": "rendered with the chat template as served", "strip": "rendered with no safety preamble"}
# a line of the training list; one whose values the recipe lacks is left out
RECIPE_LINES = (
    "- LoRA: rank {rank}, alpha {alpha}, dropout {dropout} on {target_modules}.",
    "- Schedule: {epochs} epochs ({steps} optimizer steps), peak learning rate {learning_rate:g}, batch size "
    "{batch_size} with {gradient_accumulation_steps} accumulation steps, at most {max_length} tokens per row, "
    "{dtype}, seed {seed}.",
    "- Rows: {rows} benign instruction rows ({rows_per_language}) from `sft_train/` of `{dataset}`, {rendering}.",
    "- Base model: `{base_model}`.",
)
MERGED_HOW = (
    "- Evaluated checkpoint: the base model with this adapter merged in (`merge_and_unload()` on CPU in bfloat16); "
    "the merged safetensors weights are not published. sha256 over the sha256 digests of the merged shards, in "
    "filename order: `{merged}`."
)
INTENDED = (
    "Research on how fine-tuning changes the safety behaviour of a small multilingual model, and reproduction of "
    "the tables above. It is an evaluation artifact, not a safety guarantee: measure it on your own prompts before "
    "any other use."
)
# one more paragraph on a dataset card once a full run left the safety preamble out
SECOND = {
    "eval": "`{tree}labels/<checkpoint>/` holds the labels of the runs made with no safety preamble.",
    "results": (
        "`{tree}tables/` holds the tables of the runs made with no safety preamble; their records are in `runs/` "
        "with the note `{note}`."
    ),
    "responses": (
        "`{tree}responses/` and `{tree}judge_raw/` hold the same files for the runs made with no safety preamble."
    ),
}
REPAIRED_WHAT = (
    "`{model}` after benign fine-tuning and a weight-space interpolation of the fine-tuned weights back towards "
    "the base model's."
)
REPAIRED_HOW = (
    "How it was made: the fine-tuned checkpoint was repaired in weight space with SafeTune; the strength was "
    "chosen on a small screen, and the choice with every screened value is in `alpha_screen.json` of the results "
    "dataset."
)
GGUF_RECORD = "`gguf.json` beside the file records its sha256, the release and the source revision."
GGUF_WHAT = (
    "The repaired model `{repaired}` as a q4_k_m GGUF, converted and quantized with llama.cpp `{llamacpp}`. "
    + GGUF_RECORD
)
MERGED_GGUF_WHAT = (
    "The fine-tuned model, `{model}` with the adapter `{adapter}` merged in, as a q4_k_m GGUF: converted and "
    "quantized with llama.cpp `{llamacpp}`. " + GGUF_RECORD
)
GGUF_USAGE = """```bash
hf download {repo} {file} --local-dir .
llama-server -m {file}
```"""
TITLES = {
    "eval": "Six-language safety evaluation set for Tiny Aya Global",
    "results": "Multilingual safety map of Tiny Aya Global: runs, tables, costs and logs",
    "responses": "Multilingual safety map of Tiny Aya Global: raw responses and judge outputs",
    "adapter": "Tiny Aya Global, benign multilingual fine-tune (LoRA adapter)",
    "adapter_nopre": "Tiny Aya Global, benign multilingual fine-tune with no safety preamble (LoRA adapter)",
    MERGED_GGUF: "Tiny Aya Global, benign multilingual fine-tune, q4_k_m GGUF",
    "repaired": "Tiny Aya Global, repaired after benign fine-tuning",
    "gguf": "Tiny Aya Global, repaired, q4_k_m GGUF",
}
TAGS = ["safety", "red-teaming", "multilingual"]
TREE = ids.tree("full", "strip")
# the condition each adapter was trained under
TRAINED = dict(zip(ADAPTERS, ids.PREAMBLES, strict=True))
# the trees of the other prompt conditions; each holds the folders of the main tree
OTHER_TREES = {
    TREE: "with no safety preamble",
    ids.tree("full", "patch"): "with the safety patch as the system message",
}
SMOKE_TREE = ids.tree("smoke")


def _fields(model: type[BaseModel]) -> tuple[str, ...]:
    return tuple(field.alias or name for name, field in model.model_fields.items())


_PROMPT = ("prompt_id", "language", "set", "prompt")
# folder or file, whether its first level is a checkpoint, what it holds, the fields of a row, and whether every
# publication holds it
LAYOUT = {
    "eval": (
        ("manifest.json", False, "counts, seed, pinned source revisions and source licences", (), True),
        (
            "prompts/harmful/",
            False,
            "the harmful prompts, one file per language",
            (*_PROMPT, "global_or_local", "harm_category"),
            True,
        ),
        (
            "prompts/benign_heldout/",
            False,
            "held-out benign prompts with their reference answers",
            (*_PROMPT, "target"),
            True,
        ),
        ("prompts/xstest/", False, "safe prompts that only look harmful, in English", (*_PROMPT, "type"), True),
        (
            "sft_train/",
            False,
            "the benign fine-tuning rows, each a prompt and a completion as chat turns",
            ("prompt", "completion"),
            True,
        ),
        (
            ids.tree("full", variants=True),
            False,
            "script stress sets: evaluation prompts written again in Latin script or mixed with English, each "
            "linked to its parent prompt",
            ("prompt_id", "parent_prompt_id", "language", "set", "variant", "prompt"),
            False,
        ),
        (
            "labels/",
            True,
            "the judge's labels for each evaluated checkpoint; `benign/` under a checkpoint holds the chrF of its "
            "answers to the benign prompts",
            _fields(LabelRow),
            True,
        ),
    ),
    "results": (
        ("runs/", False, "one record per job: inputs, outputs, minutes, cost and notes", (), True),
        ("tables/", False, "the result tables as JSON and as Markdown", (), True),
        ("ledger.jsonl", False, "the cost ledger, one line per job", (), True),
        ("logs/", False, "masked job logs", (), True),
    ),
    "responses": (
        (
            "responses/",
            True,
            "each checkpoint's answers, one file per prompt set and language",
            _fields(GenerationRow),
            True,
        ),
        ("judge_raw/", True, "the judge's unparsed answer to each judged response", (), True),
    ),
}


def repo_ids(settings: Settings) -> dict[str, str]:
    """Repo id per card key; the second adapter and the merged model's GGUF are named after the adapter."""
    derived = {
        ADAPTERS[1]: repo_names(settings, "full", "strip")[0],
        MERGED_GGUF: quant.merged_names(settings, "full")[1],
    }
    return {key: derived.get(key) or f"{settings.owner}/{getattr(settings.repos, key)}" for key in (*DATASETS, *MODELS)}


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


def _unlicensed(manifest: Manifest) -> list[str]:
    """Dataset sources the manifest gives no licence for."""
    return [ref.repo for ref in manifest.sources if ref.kind == "dataset" and not manifest.licences.get(ref.repo)]


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
        closed = settings.judge.closed
        named = f"{closed.name} under `{closed.rubric}`" if closed else "none configured"
        named += ", as configured; no labels tabled yet"
    else:
        judge = tables.judge
        named = f"{judge.repo} under `{judge.prompt_format}` (rubric text sha256 `{judge.revision[:12]}`)"
    return f"- Judge: {named} (its own licence or terms apply)."


def _provenance(
    settings: Settings, manifest: Manifest | None, tables: Tables | None, writeup: str | None, second: bool
) -> list[str]:
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
    condition = f"was kept in the main runs; the runs under `{TREE}` left it out" if second else f"{kept} in every run"
    return [
        "## Provenance and licences\n",
        f"- Base model: [{model}](https://huggingface.co/{model})"
        + (f" at revision `{pin}`" if pin else "")
        + f", {MODEL_LICENSE}; every model derived from it keeps that licence.",
        f"- Preamble condition: the chat template's safety preamble {condition}.",
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


def _repo(checkpoint: str) -> str | None:
    """Name of the repo a checkpoint reference is in, without its owner."""
    try:
        return ids.parse(checkpoint).repo.rpartition("/")[2]
    except ValueError:
        return None


def _rates(tables: Tables | None, column: str, settings: Settings, repo: str | None = None) -> list[str]:
    """The column's safe rates, one line per row; with ``repo`` only those of a checkpoint in that repo."""
    if tables is None or _column(tables, column)[0] not in ("tabled", "not comparable"):
        return []
    rows = [row for row in tables.safety if row.metric == "safe" and column_of(row.checkpoint, settings) == column]
    return [
        f"- {row.row}: {rate_cell(row)}"
        for row in rows
        if repo is None or _repo(row.checkpoint) == repo.rpartition("/")[2]
    ]


def _quantized(tables: Tables | None, settings: Settings, repo: str) -> list[str]:
    """What the fourth column says of the file in ``repo``."""
    rows = _rates(tables, "repaired_q4", settings, repo)
    if rows:
        return ["Quantized evaluation, as the safe rate on harmful prompts:", "\n".join(rows)]
    state = _column(tables, "repaired_q4")[0]
    if state in ("tabled", "not comparable"):
        return ["Quantized evaluation: not run (the tabled file is another repo's)."]
    kept = " The file stays published with its record." if state == "failed" else ""
    return [f"Quantized evaluation: {_stated(tables, 'repaired_q4')}.{kept}"]


def _drift_run(records: list[RunRecord], preamble: str) -> RunRecord | None:
    """The latest full drift run of a training condition that pushed merged weights."""
    drifts = [run for run in records if run.kind == "drift" and run.mode == "full" and run.merged_sha256]
    drifts = [run for run in drifts if ids.preamble_of(run.notes) == preamble]
    return max(drifts, key=lambda run: run.started, default=None)


def _fine_tuned(found: Tables | None, key: str, settings: Settings) -> str | None:
    """The checkpoint tabled beside the baseline as the fine-tuned model, when it is this adapter's."""
    if found is None or any(_column(found, column)[0] != "tabled" for column in COLUMNS[:2]):
        return None
    own = {repo.rpartition("/")[2] for repo in repo_names(settings, "full", TRAINED[key])}
    tabled = {row.checkpoint for row in found.safety if column_of(row.checkpoint, settings) == COLUMNS[1]}
    return next((checkpoint for checkpoint in sorted(tabled) if _repo(checkpoint) in own), None)


def _baseline(found: Tables, settings: Settings) -> str | None:
    tabled = (row.checkpoint for row in found.safety if column_of(row.checkpoint, settings) == COLUMNS[0])
    return next(tabled, None)


def _drop(found: Tables, checkpoint: str, settings: Settings) -> Fraction | None:
    """How far the checkpoint's pooled safe rate lies below the baseline's; None without both rates."""
    pooled = {row.checkpoint: row for row in found.safety if (row.metric, row.row) == ("safe", POOLED) and row.n}
    base, tuned = pooled.get(_baseline(found, settings)), pooled.get(checkpoint)
    # exact, as the drift flag of the table is
    return Fraction(base.k, base.n) - Fraction(tuned.k, tuned.n) if base and tuned else None


def _measured(found: Tables, checkpoint: str, settings: Settings, condition: str) -> list[str]:
    """One condition on an adapter card: its per-language table, then the changes over every language."""
    base = _baseline(found, settings)
    title, rendered, _ = CONDITIONS[condition]
    judge = f"judge={found.judge.repo.rpartition('/')[2]}"

    def rate(metric: str, name: str, of: str) -> str:
        rows = found.safety if metric != "over_refusal" else found.over_refusal
        row = next((row for row in rows if (row.metric, row.row, row.checkpoint) == (metric, name, of)), None)
        return f"{row.rate:.1%} (n={row.n})" if row and row.n else "n/a"

    def chrf(name: str, of: str) -> str:
        row = next((row for row in found.chrf if (row.row, row.checkpoint) == (name, of)), None)
        return f"{row.chrf * 100:.1f} (n={row.n})" if row and row.n else "n/a"

    def change(metric: str, name: str) -> str | None:
        wanted = (name, metric, *COLUMNS[:2])
        delta = next((delta for delta in found.deltas if (delta.row, delta.metric, delta.a, delta.b) == wanted), None)
        if delta is None or delta.delta is None:
            return None
        source = "chrF1" if metric == "chrf" else judge
        return delta_cell(delta, f"{delta.n_pairs} pairs", source, condition, WORDS[metric])

    names = [name for name in dict.fromkeys(row.row for row in found.safety) if "+" not in name]
    lines = [
        [
            name,
            *(rate(metric, name, of) for metric in ("safe", "refusal") for of in (base, checkpoint)),
            *(chrf(name, of) for of in (base, checkpoint)),
        ]
        for name in names
    ]
    table = [f"| {' | '.join(row)} |" for row in (MEASURED_HEAD, ["---"] * len(MEASURED_HEAD), *lines)]
    over = f"{rate('over_refusal', 'en', base)} for the base model, {rate('over_refusal', 'en', checkpoint)} fine-tuned"
    moved = change("over_refusal", "en")
    changes = {
        "Safe rate, all languages": change("safe", POOLED),
        "Over-refusal on XSTest (English)": f"{over}; {moved}" if moved else over,
        "chrF, all languages": change("chrf", POOLED),
    }
    return [
        f"### {title}",
        f"{rendered} The base model is `{base}`, the fine-tuned model `{checkpoint}` (the base model with this "
        "adapter merged in).",
        "\n".join(table),
        "\n".join(f"- {name}: {text}" for name, text in changes.items() if text),
    ]


def _training(recipe: Mapping[str, Any] | None, run: RunRecord | None, merged: str) -> list[str]:
    """The training list of an adapter card: its recipe, the evaluated checkpoint and the cost of the run."""
    values = {key: value for key, value in (recipe or {}).items() if value is not None}
    if isinstance(values.get("target_modules"), list):
        values["target_modules"] = ", ".join(values["target_modules"])
    if isinstance(values.get("rows_per_language"), dict):
        per_language = values["rows_per_language"].items()
        values["rows_per_language"] = ", ".join(f"{code} {count}" for code, count in per_language)
    if values.get("preamble") in RENDERING:
        values["rendering"] = RENDERING[values["preamble"]]
    lines = []
    for template in RECIPE_LINES:
        if all(name in values for _, name, _, _ in Formatter().parse(template) if name):
            lines.append(template.format_map(values))
    lines += [f"- Recipe: `{RECIPE}` in this repo.", MERGED_HOW.format(merged=merged)]
    if run and run.cost_micro_usd is not None and run.minutes is not None:
        cost = f"${run.cost_micro_usd / 1_000_000:.2f} for {run.minutes} minutes on {run.flavor}"
        lines.append(f"- Cost: {cost} (`runs/{run.run_id}.json` in the results dataset).")
    return lines


def _limits(settings: Settings, manifest: Manifest | None) -> list[str]:
    counts = sorted({count.harmful for count in manifest.per_language.values()}) if manifest else []
    span = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]} to {counts[-1]}" if counts else None
    return [
        "- One judge model under one rubric wrote every label; both are named below.",
        *([f"- {span} harmful prompts per language: read the row over all languages first."] if span else []),
        f"- The prompts are plain requests from `{settings.sources.harmful}`; jailbreak wrappers were not tested.",
        f"- Answers were generated with greedy decoding, at most {settings.decoding.max_new_tokens} new tokens.",
        f"- One training run with one seed made this adapter; it was trained and measured in "
        f"{', '.join(settings.languages)} only.",
    ]


def _comparisons(
    key: str, settings: Settings, tables: Tables | None, nopreamble: Tables | None
) -> list[tuple[Fraction, str, Tables, str]]:
    """Per condition whose table holds the adapter's checkpoint: its drop, the condition, the table, the checkpoint."""
    measured = []
    for found, condition in ((tables, "keep"), (nopreamble, "strip")):
        checkpoint = _fine_tuned(found, key, settings)
        drop = _drop(found, checkpoint, settings) if checkpoint else None
        if drop is not None:
            measured.append((drop, condition, found, checkpoint))
    return measured


def _notice(measured: list[tuple[Fraction, str, Tables, str]], cleared: str, whose: str, this: str) -> list[str]:
    """The opening of a card of fine-tuned weights: not for deployment unless a table shows them not less safe."""
    worst = max(measured, key=lambda found: found[0], default=None)
    if worst is None:
        return [f"**{DO_NOT_DEPLOY}** {UNMEASURED}"]
    if worst[0] * 100 > DRIFT_POINTS:
        less = LESS_SAFE.format(whose=whose, points=float(worst[0] * 100), condition=CONDITIONS[worst[1]][2])
        return [f"**{DO_NOT_DEPLOY}** {less}", LSAL.format(this=this)]
    return [f"**{cleared}** {MEASUREMENTS}"]


def _adapter(
    key: str,
    settings: Settings,
    manifest: Manifest | None,
    tables: Tables | None,
    nopreamble: Tables | None,
    records: list[RunRecord],
    recipe: Mapping[str, Any] | None,
) -> list[str]:
    """An adapter card between its title and its provenance; its notice follows what its tables measured."""
    preamble = TRAINED[key]
    names = {"model": settings.sources.model, "languages": ", ".join(settings.languages), "tree": TREE}
    measured = _comparisons(key, settings, tables, nopreamble)
    notice = _notice(measured, f"{RESEARCH} {NOT_CERTIFIED}", "Its", "This adapter makes")
    results = [HOW_TO_READ.format(results=repo_ids(settings)["results"])] if measured else [NOT_MEASURED]
    for _, condition, found, checkpoint in measured:
        results += _measured(found, checkpoint, settings, condition)
    by_id = {run.run_id: run for run in records if run.kind == "drift"}
    latest = _drift_run(records, preamble)
    run = by_id.get((recipe or {}).get("run_id")) or latest
    pins = _pins()
    pinned = [f"{name} {pins[name]}" for name in ("transformers", "peft") if name in pins]
    tokens = settings.decoding.max_new_tokens
    usage = USAGE.format(model=settings.sources.model, repo=repo_ids(settings)[key], tokens=tokens)
    return [
        *notice,
        (ADAPTER_NOPRE_WHAT if preamble == "strip" else ADAPTER_WHAT).format_map(names),
        "## Results",
        *results,
        "## How to use",
        usage,
        *([f"The project pins {' and '.join(pinned)}."] if pinned else []),
        *([USAGE_NOPRE] if preamble == "strip" else []),
        "## Training",
        "\n".join(_training(recipe, run, latest.merged_sha256 if latest else "not recorded")),
        "## Intended use and limits",
        INTENDED,
        "\n".join(_limits(settings, manifest)),
    ]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'s' * (count != 1)}"


def _files(key: str, paths: Sequence[str] | None) -> list[str]:
    """The folders of a dataset as a list; with the repo's file list, only the ones it holds, each with its counts."""
    main = None if paths is None else [path for path in paths if not path.startswith((SMOKE_TREE, *OTHER_TREES))]
    lines = []
    for place, by_checkpoint, what, fields, always in LAYOUT[key]:
        under = [path for path in main or () if path.startswith(place)] if place.endswith("/") else []
        if main is None and not always or main is not None and not under and place not in main:
            continue
        checkpoints = {path.split("/")[place.count("/")] for path in under}
        counts = [_plural(len(checkpoints), "checkpoint")] * by_checkpoint + [_plural(len(under), "file")]
        count = f" ({', '.join(counts)})" if under else ""
        shown = f" Fields: {', '.join(f'`{name}`' for name in fields)}." if fields else ""
        lines.append(f"- `{place}`{count}: {what}.{shown}")
    for tree, condition in OTHER_TREES.items():
        under = [path for path in paths or () if path.startswith(tree)]
        if under:
            lines.append(f"- `{tree}` ({_plural(len(under), 'file')}): the same folders for the runs made {condition}.")
    return ["## Files", "\n".join(lines)] if lines else []


def _bodies(
    settings: Settings,
    manifest: Manifest | None,
    tables: Tables | None,
    records: list[RunRecord],
    nopreamble: Tables | None,
    second: bool,
    recipes: Mapping[str, Mapping[str, Any] | None],
    files: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Paragraphs of each card, between its title and its provenance list."""
    sources, repos = settings.sources, repo_ids(settings)
    names = {
        "model": sources.model,
        "harmful": sources.harmful,
        "benign": sources.benign,
        "xstest": sources.xstest,
        "languages": ", ".join(settings.languages),
        "repaired": repos["repaired"],
        "adapter": repos["adapter"],
        "llamacpp": settings.llamacpp_release.tag,
        "tree": TREE,
        "note": ids.preamble_note("strip"),
    }
    counts = [
        f"| {code} | {count.harmful} | {count.benign_heldout} | {count.benign_train} |"
        for code, count in (manifest.per_language.items() if manifest else ())
    ]
    table = ["| language | harmful | benign held-out | benign train |", "|---|---|---|---|", *counts]
    residual = _rates(tables, "repaired", settings)

    def filled(*templates: str) -> list[str]:
        return [template.format_map(names) for template in templates]

    def also(key: str) -> list[str]:
        return filled(SECOND[key]) if second else []

    def states(found: Tables | None) -> str:
        return "\n".join(f"- {column}: {_stated(found, column)}" for column in COLUMNS)

    def served(key: str) -> list[str]:
        usage = GGUF_USAGE.format(repo=repos[key], file=quant.GGUF_FILE)
        return [*_quantized(tables, settings, repos[key]), "## How to use", usage]

    stripped = ["State of each planned column with no safety preamble:", states(nopreamble)] if nopreamble else []
    bodies = {
        "eval": [
            *filled(EVAL_NOTICE, EVAL_WHAT),
            *also("eval"),
            *filled(EVAL_HOW),
            RUBRIC,
            *(["\n".join(table)] if counts else []),
            *_files("eval", files.get("eval")),
        ],
        "results": [
            *filled(RESULTS_WHAT),
            *also("results"),
            *filled(RESULTS_HOW),
            f"State of each planned column. {COLUMN_NAMES}",
            states(tables),
            *stripped,
            *_files("results", files.get("results")),
        ],
        "responses": [
            *filled(RESPONSES_WHAT),
            *also("responses"),
            *filled(RESPONSES_HOW),
            RUBRIC,
            *_files("responses", files.get("responses")),
        ],
        # the file is the first adapter's merged model: it opens as that adapter's measurement lets it
        MERGED_GGUF: [
            *_notice(
                _comparisons(ADAPTERS[0], settings, tables, nopreamble),
                NOT_CERTIFIED,
                "Its unquantized model's",
                "This file is",
            ),
            *filled(MERGED_GGUF_WHAT),
            *served(MERGED_GGUF),
        ],
        "repaired": [
            f"**{NOT_CERTIFIED}** {MEASUREMENTS}",
            *([NOT_SERVED] if settings.publish.space_not_serving else []),
            *filled(REPAIRED_WHAT, REPAIRED_HOW),
            *(
                ["Residual failures, as the safe rate on harmful prompts:", "\n".join(residual)]
                if residual
                else [f"Residual failures: {_stated(tables, 'repaired')}."]
            ),
        ],
        "gguf": [f"**{NOT_CERTIFIED}**", *filled(GGUF_WHAT), *served("gguf")],
    }
    for key in ADAPTERS:
        bodies[key] = _adapter(key, settings, manifest, tables, nopreamble, records, recipes.get(key))
    return bodies


def render_cards(
    settings: Settings,
    *,
    manifest: Manifest | None = None,
    tables: Tables | None = None,
    records: Iterable[RunRecord] = (),
    writeup: str | None = None,
    nopreamble: Tables | None = None,
    recipes: Mapping[str, Mapping[str, Any] | None] | None = None,
    files: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """README text per repo id for every repo a publication may open; ``writeup`` defaults to the code repo's article.

    ``nopreamble`` is the table of the runs with no safety preamble. The cards name that condition once it is
    given, or a full run made under it is among ``records``. ``recipes`` holds, per adapter key, the recipe file
    of its repo; ``files`` holds, per dataset key, the paths its repo will hold, which its card then counts.
    """
    records = list(records)
    second = nopreamble is not None or any(
        run.mode == "full" and ids.preamble_of(run.notes) == "strip" for run in records
    )
    rendered, repos = {}, repo_ids(settings)
    bodies = _bodies(settings, manifest, tables, records, nopreamble, second, recipes or {}, files or {})
    for key in (*DATASETS, *MODELS):
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
                library_name={**dict.fromkeys(ADAPTERS, "peft"), "repaired": "transformers"}.get(key),
                language=list(settings.languages),
                tags=[*TAGS, "gguf"] if key in (MERGED_GGUF, "gguf") else TAGS,
            )
            top = []
        # the second adapter exists only through a run of the second condition
        provenance = _provenance(settings, manifest, tables, writeup, second or key == ADAPTERS[1])
        blocks = [*top, f"# {TITLES[key]}", *bodies[key], "\n".join(provenance)]
        rendered[repos[key]] = f"---\n{data.to_yaml()}\n---\n\n" + "\n\n".join(blocks) + "\n"
    return rendered


def missing(key: str, text: str | None, settings: Settings, manifest: Manifest | None = None) -> list[str]:
    """What the card of a repo lacks before it may be public; empty when it is complete."""
    if text is None:
        return ["README.md"]
    card = RepoCard(text, ignore_metadata_errors=True)
    data, wanted = card.data.to_dict(), license_of(key, manifest)
    lacks = []
    unlicensed = [] if manifest is None else _unlicensed(manifest)
    if key == "eval" and (manifest is None or unlicensed):
        named = "no manifest.json" if manifest is None else f"none in manifest.json for {', '.join(unlicensed)}"
        lacks.append(f"the source licences ({named})")
    elif data.get("license") != wanted:
        shown = ", ".join(wanted) if isinstance(wanted, list) else wanted
        lacks.append(f"license: {shown} in the card metadata")
    if key in MODELS and data.get("base_model") != settings.sources.model:
        lacks.append(f"base_model: {settings.sources.model} in the card metadata")
    if key in ("results", "responses") and not card.text.lstrip().startswith(WARNING):
        lacks.append("the content warning as the first line")
    # what a card must hold: any one of the texts named
    needles = {
        "the harmful-prompts notice": [f"harmful prompts from `{settings.sources.harmful}`"] if key == "eval" else [],
        # fine-tuned weights say either that they are not for deployment or, once measured, not certified for it
        "the deployment notice": [DO_NOT_DEPLOY, NOT_CERTIFIED] if key in FINE_TUNED else [],
        "the not-certified notice": [NOT_CERTIFIED] if key in MODELS and key not in FINE_TUNED else [],
        "the base model's licence": [MODEL_LICENSE] if key in MODELS else [],
        "the link to the code": [settings.code.origin],
        "the link to the write-up": ["- Write-up: http"],
    }
    return lacks + [name for name, any_of in needles.items() if any_of and not any(n in card.text for n in any_of)]
