"""Build the data file of the results Space from the published label files and tables.

    python space/build_data.py                                  # reads the datasets from the Hub
    python space/build_data.py --eval-dir DIR --tables-dir DIR  # reads local copies
    python space/build_data.py --eval-dir DIR --responses-dir DIR   # adds the answer-length counts and the explorer
    python space/build_data.py --eval-dir DIR --choice-dir DIR      # adds the question pairs from their choice table
    python space/build_data.py --eval-dir DIR --choice-dir DIR --choice-eval-dir DIR --choice-responses-dir DIR
                                                                    # and the sample item, from prompts and answers
    python space/build_data.py --no-question-pairs                  # a page without the question pairs

The three datasets are the ones ``driftmap.toml`` names. The served template and no safety preamble, each with
the base model and the fine-tuned model, are always read.
The overfit fine-tune, the safety-patch condition and the script variants are read when their label files exist,
and the safety patch on the held-out prompts when its table is among the tables.
The data file holds counts and intervals only: no prompt, no response and no per-prompt row.

With the responses at hand the script also writes ``explorer.json`` beside the data file, for the page's replay
explorer: a capped set of harmful-set prompts with their text and what each measured system answered. An answer
judged harmful, or left without a verdict, is listed without its text unless ``--include-harmful`` is given,
and then keeps only its first characters. Prompts of the non-consensual sexual content category are left out.
That file is a build output.

The question pairs (who did it, when the text does not say) are read from their choice table: counts, intervals,
the count under the strict reading and the paired changes the table states. With the choice prompts and the
stored answers at hand the script also writes
``samples.json``: one question pair from BBQ in two languages with what the base model and the fine-tune answered.
"""

import argparse
import json
import math
import shutil
import statistics
import sys
import tomllib
from collections import Counter
from pathlib import Path

SCHEMA = "driftmap.space/7"
CONFIG = Path(__file__).resolve().parent.parent / "driftmap.toml"


def repos(config: Path = CONFIG) -> tuple[str, str, str]:
    """The evaluation, results and responses datasets the project configuration names, each as ``owner/name``."""
    settings = tomllib.loads(config.read_text(encoding="utf-8"))
    return tuple(f"{settings['owner']}/{settings['repos'][key]}" for key in ("eval", "results", "responses"))


EVAL_REPO, RESULTS_REPO, RESPONSES_REPO = repos()
# id, name on the page, folder of the label files in the evaluation dataset, tables file in the results dataset
CONDITIONS = (
    ("served", "served template", "labels", "tables/tables.json"),
    ("nopreamble", "no safety preamble", "nopreamble/labels", "nopreamble/tables/tables-nopreamble.json"),
    ("patched", "safety patch", "patched/labels", "patched/tables/tables-patched.json"),
)
# the conditions every build needs; another one is read for the models that have label files in it
REQUIRED = ("served", "nopreamble")
# id, name on the page, folder under each label tree
MODELS = (
    ("base", "base model", "tiny-aya-global--af89d219b53e"),
    ("finetuned", "fine-tuned model", "tiny-aya-global-drift-merged--daccc14f6c19"),
)
# a second fine-tune of the same rows: id, name, label folder, and its tables file in each condition it is
# evaluated in, the condition of its repair first
OVERFIT = (
    "overfit",
    "overfit fine-tune",
    "tiny-aya-global-drift-merged-nopre--d845b23452bb",
    {
        "nopreamble": "nopreamble/tables/tables-nopreamble-aggressive.json",
        "served": "tables/tables-aggressive.json",
    },
)
# that fine-tune after its weights are interpolated with the base model's: id, name, condition, label folder;
# its paired changes are the drifted-to-repaired columns of the overfit fine-tune's tables file in that condition
REPAIRED = ("repaired", "repaired fine-tune", "nopreamble", "tiny-aya-global-repaired-nopre--ba1563a2e8c8")
# the script variants of the harmful prompts, answered under the served template: prompt files, label tree
VARIANT_PROMPTS = "variants/prompts/harmful"
VARIANT_LABELS = "variants/labels"
EXPLORER_SCHEMA = "driftmap.space.explorer/1"
# the safety patch on harmful prompts held out of the evaluation set: its table in the results dataset
HELDOUT_TABLE = "tables/heldout.json"
HELDOUT_METRICS = ("harmful", "refusal")
# the question pairs: their table in the results dataset, their prompts and the stored answers to them
CHOICE_TABLE = "tables/choice-defense.json"
CHOICE_PROMPTS = "choice/prompts/benign/{language}.jsonl"
# both sample arms answer under the served template, whose tree is the root of the responses dataset
CHOICE_ANSWERS = "choice/responses/{slug}/benign/{language}.jsonl"
CHOICE_METRICS = ("person_pick", "accuracy", "twin")
# the arms every other arm is set against, and the two whose answers the samples show
CHOICE_BASE, CHOICE_TUNED = "baseline/keep", "drifted/keep"
SAMPLES_SCHEMA = "driftmap.space.samples/1"
# the question pair the page shows as it was asked without options, and in which languages
SAMPLE_ITEM = "bbq.Gender_identity.3448"
SAMPLE_LANGUAGES = ("en", "ru")
# the systems the explorer replays, in the order of its columns: condition, model, name on the page, note
SYSTEMS = (
    ("served", "base", "Base model", None),
    ("patched", "base", "Base model + safety patch", None),
    ("served", "finetuned", "Light fine-tune", None),
    ("patched", "finetuned", "Light fine-tune + safety patch", None),
    ("nopreamble", "overfit", "Overfit fine-tune", "research artifact, not for deployment"),
    ("nopreamble", "repaired", "Repaired fine-tune", None),
)
# the explorer shows prompt text; prompts of this category are left out of its file
WITHHELD_CATEGORY = "Non-consensual sexual content"
# prompts the explorer file holds at most
EXPLORER_CAP = 300
# characters kept of an answer judged harmful or left without a verdict, and of any other answer
HARMFUL_CUT = 300
ANSWER_CUT = 1500
LANGUAGES = ("en", "fr", "es", "ru", "hi", "ar")
SETS = ("harmful", "xstest")
LABELLED = ("yes", "no")
Z95 = 1.959963984540054


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; split on newlines only, since a prompt may hold other line separators."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def verdict(row: dict) -> str:
    """One word per judged pair: harmful, refused, answered (not harmful, no refusal) or unscored."""
    if row["response_harm"] not in LABELLED or row["refusal"] not in LABELLED:
        return "unscored"
    if row["response_harm"] == "yes":
        return "harmful"
    return "refused" if row["refusal"] == "yes" else "answered"


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """95 % Wilson score interval for k of n; None when n is 0."""
    if n == 0:
        return None
    share, scale = k / n, 1 + Z95**2 / n
    centre = (share + Z95**2 / (2 * n)) / scale
    half = Z95 * math.sqrt(share * (1 - share) / n + Z95**2 / (4 * n**2)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(up: int, down: int) -> float:
    """Exact two-sided sign test on ``up`` changes one way and ``down`` the other; 1.0 without a change."""
    total = up + down
    tail = sum(math.comb(total, i) for i in range(min(up, down) + 1)) / 2**total
    return min(1.0, 2 * tail)


def _rate(k: int, n: int) -> list:
    interval = wilson(k, n)
    # six digits, so a bound shown as a percentage with one decimal rounds as the tables round it
    return [k, n, *(round(bound, 6) for bound in interval)] if interval else [k, n, None, None]


def _prompts(eval_dir: Path) -> dict[str, dict]:
    """Language and harm categories of every harmful prompt, by prompt id; the text is dropped here."""
    found = {}
    for path in sorted((eval_dir / "prompts" / "harmful").glob("*.jsonl")):
        for row in read_jsonl(path):
            categories = sorted(json.loads(row["harm_category"]))
            found[row["prompt_id"]] = {"language": row["language"], "categories": categories}
    return found


def _has(eval_dir: Path, tree: str, folder: str) -> bool:
    return any((eval_dir / tree / folder).glob("*.jsonl"))


def _labels(eval_dir: Path, tree: str, folder: str) -> tuple[list[dict], list[float]]:
    """Judged rows of one checkpoint in one condition, and the chrF of its benign answers."""
    here = eval_dir / tree / folder
    judged = [row for path in sorted(here.glob("*.jsonl")) for row in read_jsonl(path)]
    scores = [row["chrf"] for path in sorted(here.glob("benign/*.jsonl")) for row in read_jsonl(path)]
    if not judged:
        raise FileNotFoundError(f"no label files under {tree}/{folder} in {eval_dir}")
    return judged, [score for score in scores if score is not None]


def _summary(judged: list[dict], scores: list[float]) -> dict:
    scored = {name: [row for row in judged if row["set"] == name and verdict(row) != "unscored"] for name in SETS}
    harmful, xstest = scored["harmful"], scored["xstest"]
    return {
        "harmful": _rate(sum(row["response_harm"] == "yes" for row in harmful), len(harmful)),
        "refusal": _rate(sum(row["refusal"] == "yes" for row in harmful), len(harmful)),
        # over-refusal is the refusal rate on prompts that are safe to answer
        "over_refusal": _rate(sum(row["refusal"] == "yes" for row in xstest), len(xstest)),
        "chrf": [round(sum(scores) / len(scores), 4), len(scores)] if scores else [None, 0],
    }


def _cells(condition: str, model: str, verdicts: dict[str, str], prompts: dict[str, dict], categories: list) -> list:
    """Harmful answers of scored pairs for every language and category, with ``all`` as the marginal of each."""
    counts, sizes = Counter(), Counter()
    for prompt_id, word in verdicts.items():
        if word == "unscored":
            continue
        prompt = prompts[prompt_id]
        for language in (prompt["language"], "all"):
            for category in (*prompt["categories"], "all"):
                sizes[language, category] += 1
                counts[language, category] += word == "harmful"
    cells = []
    for language in (*LANGUAGES, "all"):
        for category in (*categories, "all"):
            k, n = counts[language, category], sizes[language, category]
            low, high = _rate(k, n)[2:]
            row = {"condition": condition, "model": model, "language": language, "category": category}
            cells.append(row | {"k": k, "n": n, "lo": low, "hi": high})
    return cells


def _flips(before: dict[str, str], after: dict[str, str]) -> dict[str, int]:
    pairs = [(before[key], after[key]) for key in before.keys() & after.keys()]
    pairs = [(first == "harmful", second == "harmful") for first, second in pairs if "unscored" not in (first, second)]
    return {
        "pairs": len(pairs),
        "harmful_both": pairs.count((True, True)),
        "harmful_to_not": pairs.count((True, False)),
        "not_to_harmful": pairs.count((False, True)),
    }


def _both(before: dict[str, dict], after: dict[str, dict]) -> list[str]:
    """The keys labelled on both sides, in order."""
    shared = sorted(before.keys() & after.keys())
    return [key for key in shared if "unscored" not in (verdict(before[key]), verdict(after[key]))]


def _paired(before: dict[str, dict], after: dict[str, dict]) -> dict:
    """Counts over the keys labelled on both sides: harmful answers, refusals and plain answers before and after,
    the moves between harmful and not, and the sign test on those moves."""
    keys = _both(before, after)
    sides = ([before[key] for key in keys], [after[key] for key in keys])
    flips = _flips(*({key: verdict(row) for key, row in zip(keys, rows, strict=True)} for rows in sides))
    return flips | {
        "harmful": [sum(row["response_harm"] == "yes" for row in rows) for rows in sides],
        "refusal": [sum(row["refusal"] == "yes" for row in rows) for rows in sides],
        "answered": [sum(verdict(row) == "answered" for row in rows) for rows in sides],
        "p": sign_test(flips["not_to_harmful"], flips["harmful_to_not"]),
    }


def _variants(eval_dir: Path, served: dict[str, dict[str, dict]]) -> list[dict]:
    """One row per script variant and model: the variant prompts against their parents under the served template."""
    found = []
    for path in sorted((eval_dir / VARIANT_PROMPTS).glob("*.jsonl")):
        prompts = read_jsonl(path)
        for model, _, folder in MODELS:
            labels = eval_dir / VARIANT_LABELS / folder / path.name
            if not prompts or not labels.is_file():
                continue
            answers = {row["prompt_id"]: row for row in read_jsonl(labels)}
            # a parent's row stands under its variant's id, so the two sides pair up
            parents = {row["prompt_id"]: served[model].get(row["parent_prompt_id"]) for row in prompts}
            parents = {key: row for key, row in parents.items() if row}
            # as the variant tables do: a variant the judge read as another request than its parent is left out
            tied = [(key, parents[key]["request_harm"], row["request_harm"]) for key, row in answers.items() if key in parents]
            flipped = [key for key, old, new in tied if old in LABELLED and new in LABELLED and old != new]
            answers = {key: row for key, row in answers.items() if key not in flipped}
            both = _both(parents, answers)
            # how often the judge reads the request as harmful on each side: a check that a variant kept its request
            asked = [sum(rows[key]["request_harm"] == "yes" for key in both) for rows in (parents, answers)]
            head = {"code": path.stem, "parent_language": prompts[0]["parent_language"], "kind": prompts[0]["variant"]}
            counts = _paired(parents, answers) | {"request_harm": asked, "left_out": len(flipped)}
            found.append(head | {"model": model, "prompts": len(prompts)} | counts)
    return found


def _length(
    responses_dir: Path, tree: str, folder: str, verdicts: dict[str, str], required: bool = True
) -> dict | None:
    """Answers cut off at the token limit, and harmful answers among the complete and the cut-off ones.

    None when the checkpoint has no response files and they are not ``required``.
    """
    here = responses_dir / tree.replace("labels", "responses") / folder / "harmful"
    rows = [row for path in sorted(here.glob("*.jsonl")) for row in read_jsonl(path)]
    if not rows and not required:
        return None
    if not rows:
        raise FileNotFoundError(f"no response files under {here}")
    harmful, sizes = Counter(), Counter()
    for row in rows:
        word = verdicts.get(row["prompt_id"], "unscored")
        if word != "unscored":
            cut = row["stop_reason"] == "length"
            sizes[cut] += 1
            harmful[cut] += word == "harmful"
    return {
        "cut_off": [sum(row["stop_reason"] == "length" for row in rows), len(rows)],
        "median_tokens": statistics.median(row["generated_tokens"] for row in rows),
        "harmful_complete": [harmful[False], sizes[False]],
        "harmful_cut_off": [harmful[True], sizes[True]],
    }


def _replayed(labels: Path, responses: Path, include_harmful: bool) -> dict[str, dict] | None:
    """Per prompt id of a label folder's harmful rows: the verdict, whether the answer hit the token limit, and
    its text. None when the labels or the response files are absent."""
    answers = {row["prompt_id"]: row for path in sorted(responses.glob("*.jsonl")) for row in read_jsonl(path)}
    judged = [row for path in sorted(labels.glob("*.jsonl")) for row in read_jsonl(path) if row["set"] == "harmful"]
    if not answers or not judged:
        return None
    found = {}
    for row in judged:
        answer = answers.get(row["prompt_id"])
        if answer is None:
            continue
        word, text = verdict(row), (answer["response"] or "").strip()
        guarded = word in ("harmful", "unscored")
        cut = HARMFUL_CUT if guarded else ANSWER_CUT
        withheld = guarded and not include_harmful
        shown = None if withheld else text[:cut]
        entry = {"verdict": word, "cut_off": answer["stop_reason"] == "length", "text": shown}
        found[row["prompt_id"]] = entry | {"clipped": not withheld and len(text) > cut}
    return found


def _balanced(pools: list[list[str]], languages: dict[str, str], cap: int) -> list[str]:
    """Up to ``cap`` prompt ids, each pool used up before the next; inside a pool the language with the fewest
    prompts chosen so far gives its next one, so the choice stays spread over the languages."""
    chosen, taken = [], Counter()
    for pool in pools:
        queues = {code: [key for key in pool if languages[key] == code] for code in LANGUAGES}
        while len(chosen) < cap and any(queues.values()):
            # min keeps the first of equals, so a tie goes to the earlier language
            code = min((code for code in LANGUAGES if queues[code]), key=taken.__getitem__)
            chosen.append(queues[code].pop(0))
            taken[code] += 1
    return chosen


def explorer(
    eval_dir: str | Path, responses_dir: str | Path, cap: int = EXPLORER_CAP, include_harmful: bool = False
) -> dict:
    """The replay explorer's data: for at most ``cap`` harmful-set prompts, the prompt text and what each system
    that has labels and responses did with it; script variants hang under their parent prompt.

    Prompts where the systems disagree on a harmful answer are taken first, then those every system answers
    harmfully, then the rest; each group is spread over the languages. Prompts of the withheld category are not
    taken. The text of answers judged harmful or left without a verdict is kept only with ``include_harmful``.
    """
    eval_dir, responses_dir = Path(eval_dir), Path(responses_dir)
    trees = {key: (name, tree) for key, name, tree, _ in CONDITIONS}
    folders = {key: folder for key, _, folder in MODELS} | {OVERFIT[0]: OVERFIT[2], REPAIRED[0]: REPAIRED[3]}
    texts = {}
    for path in sorted((eval_dir / "prompts" / "harmful").glob("*.jsonl")):
        for row in read_jsonl(path):
            categories = sorted(json.loads(row["harm_category"]))
            texts[row["prompt_id"]] = {"language": row["language"], "categories": categories, "text": row["prompt"]}
    systems, replayed, rewritten = [], {}, {}
    for condition, model, name, note in SYSTEMS:
        label_tree, folder = trees[condition][1], folders[model]
        answers = responses_dir / label_tree.replace("labels", "responses") / folder / "harmful"
        found = _replayed(eval_dir / label_tree / folder, answers, include_harmful)
        if found is None:
            continue
        system = f"{condition}/{model}"
        entry = {"id": system, "name": name, "condition": trees[condition][0]}
        systems.append(entry | ({"note": note} if note else {}))
        replayed[system] = found
        if condition == CONDITIONS[0][0]:
            # the script variants are answered under the served template
            answers = responses_dir / "variants" / "responses" / folder / "harmful"
            rewritten[system] = _replayed(eval_dir / VARIANT_LABELS / folder, answers, include_harmful) or {}
    variants = {key: [] for key in texts}
    for path in sorted((eval_dir / VARIANT_PROMPTS).glob("*.jsonl")):
        for row in read_jsonl(path):
            mine = {system: found[row["prompt_id"]] for system, found in rewritten.items() if row["prompt_id"] in found}
            if mine and row["parent_prompt_id"] in variants:
                entry = {"code": path.stem, "kind": row["variant"], "text": row["prompt"], "answers": mine}
                variants[row["parent_prompt_id"]].append(entry)
    # persistent: the prompts every system with a verdict answers harmfully
    prompts, persistent = {}, []
    shown = [key for key in sorted(texts) if WITHHELD_CATEGORY not in texts[key]["categories"]]
    for key in shown:
        answers = {system: found[key] for system, found in replayed.items() if key in found}
        parts = (answers, *(entry["answers"] for entry in variants[key]))
        seen = [answer["verdict"] for part in parts for answer in part.values()]
        harmful = {word == "harmful" for word in seen if word != "unscored"}
        prompts[key] = texts[key] | {"disagree": len(harmful) == 2, "answers": answers, "variants": variants[key]}
        if harmful == {True}:
            persistent.append(key)
    disagreeing = [key for key in prompts if prompts[key]["disagree"]]
    others = [key for key in prompts if not prompts[key]["disagree"] and key not in persistent]
    pools = [disagreeing, persistent, others]
    chosen = _balanced(pools, {key: row["language"] for key, row in prompts.items()}, cap)
    short = {key: key[:12] for key in chosen}
    if len(set(short.values())) != len(short):
        raise ValueError("two chosen prompts share the first 12 characters of their id")
    order = sorted(chosen, key=lambda key: (LANGUAGES.index(prompts[key]["language"]), key))
    return {
        "schema": EXPLORER_SCHEMA,
        "cut": HARMFUL_CUT,
        "harmful_text": include_harmful,
        "total": len(texts),
        "left_out": {"category": WITHHELD_CATEGORY, "prompts": len(texts) - len(shown)},
        "disagreeing": len(disagreeing),
        "persistent": len(persistent),
        "systems": systems,
        "prompts": [{"id": short[key]} | prompts[key] for key in order],
    }


def _place(directory: Path, file: str) -> Path | None:
    """``file`` under a copy of a results dataset, or by its name in a flat folder; None when it is in neither."""
    places = (directory / file, directory / Path(file).name)
    return next((place for place in places if place.is_file()), None)


def attribution(choice_dir: str | Path) -> dict | None:
    """The question-pair result from its choice table under ``choice_dir``; None without the table.

    Per arm, scope and metric the count with its Wilson bounds and the count under the strict reading, which
    leaves out the answers read through a letter written in the local script; and the paired changes from the
    base model that the table states over all rows. An arm that adds the instruction to a fine-tune is left out.
    """
    path = _place(Path(choice_dir), CHOICE_TABLE)
    if path is None:
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    arms = [arm for arm in table["arms"] if arm["column"] == "baseline" or not arm["tag"]]
    shown = {arm["arm"] for arm in arms}
    rates = [
        {key: rate[key] for key in ("arm", "scope", "metric")}
        | dict(zip(("k", "n", "lo", "hi"), _rate(rate["k"], rate["n"])))
        | {"strict": rate["k"] - rate["k_named"]}
        for rate in table["rates"]
        if rate["arm"] in shown and rate["metric"] in CHOICE_METRICS
    ]
    changes = [
        {"arm": delta["b"], "scope": delta["scope"], "metric": delta["metric"], "delta": delta["delta"]}
        | {"ci": delta["ci"], "p": delta["p_value"], "items": delta["n_items"]}
        for delta in table["deltas"]
        if delta["basis"] == "all" and delta["a"] == CHOICE_BASE and delta["b"] in shown and delta["metric"] in CHOICE_METRICS
    ]
    return {
        "arms": [{"id": arm["arm"], "name": arm["name"], "fine_tune": arm["column"] != "baseline"} for arm in arms],
        "base": CHOICE_BASE,
        "tuned": CHOICE_TUNED,
        "scopes": table["scopes"],
        "rates": rates,
        "changes": changes,
    }


def heldout(tables_dir: str | Path) -> dict | None:
    """The safety patch on the held-out harmful prompts, from their table under ``tables_dir``; None without it.

    Per scope, a language or all of them, the harmful answers and the refusals of the base model without and
    with the patch, each count with its Wilson bounds, and the paired change the table states. Where the table
    also holds the evaluation prompts under the two conditions, ``gap`` is the held-out change minus the change
    over them, with its interval; None otherwise.
    """
    path = _place(Path(tables_dir), HELDOUT_TABLE)
    if path is None:
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    rates = {
        (rate["scope"], rate["metric"], rate["condition"]): _rate(rate["k"], rate["n"]) for rate in table["rates"]
    }
    moved = {(change["scope"], change["metric"]): change for change in table["changes"]}
    gaps = {(gap["scope"], gap["metric"]): gap for gap in (table.get("in_sample") or {}).get("gaps", ())}
    rows = []
    for scope in table["scopes"]:
        row = {"scope": scope}
        for metric in HELDOUT_METRICS:
            change = moved[scope, metric]
            row[metric] = {"before": rates[scope, metric, "keep"], "after": rates[scope, metric, "patch"]}
            row[metric] |= {"pairs": change["n_pairs"], "delta": change["delta"]}
            row[metric] |= {"ci": change["ci"], "p": change["p_value"]}
            gap = gaps.get((scope, metric))
            row[metric]["gap"] = {"delta": gap["delta"], "ci": gap["ci"]} if gap else None
        rows.append(row)
    return {"prompts": table["prompts"], "rows": rows}


def _slug(checkpoint: str) -> str:
    """The folder of a checkpoint's rows: its repo name and the first 12 characters of its commit."""
    repo, _, revision = checkpoint.partition("@")
    return f"{repo.rpartition('/')[2]}--{revision[:12]}"


def samples(choice_dir: str | Path, prompts_dir: str | Path, responses_dir: str | Path) -> dict | None:
    """The sample question pair in each sample language, as asked without options.

    ``choice_dir`` holds the choice table, ``prompts_dir`` a copy of the dataset with the choice prompts and
    ``responses_dir`` one of the dataset with the stored answers; None when the table, a prompt file or an answer
    file of the two sample arms is missing. Each sample holds the ambiguous question with the base model's and the
    fine-tune's answer, and the twin with the fine-tune's answer. Raises ValueError when the base model does not
    name one of the two people on the ambiguous question, or the fine-tune names one: such an item does not show
    the finding.
    """
    table = _place(Path(choice_dir), CHOICE_TABLE)
    if table is None:
        return None
    checkpoints = {arm["arm"]: arm["checkpoint"] for arm in json.loads(table.read_text(encoding="utf-8"))["arms"]}
    found = []
    for language in SAMPLE_LANGUAGES:
        asked = Path(prompts_dir) / CHOICE_PROMPTS.format(language=language)
        stored = [
            Path(responses_dir) / CHOICE_ANSWERS.format(slug=_slug(checkpoints[arm]), language=language)
            for arm in (CHOICE_BASE, CHOICE_TUNED)
        ]
        if not all(path.is_file() for path in (asked, *stored)):
            return None
        open_form = [row for row in read_jsonl(asked) if row["base_item"] == SAMPLE_ITEM and row["format"] == "free_form"]
        rows = {row["condition"]: row for row in open_form}
        base, tuned = ({row["prompt_id"]: row["response"] for row in read_jsonl(path)} for path in stored)
        ids = [rows[condition]["prompt_id"] for condition in ("ambiguous", "disambiguated") if condition in rows]
        if len(ids) != 2 or any(key not in side for key in ids for side in (base, tuned)):
            raise ValueError(f"{SAMPLE_ITEM} has no stored answer of both arms to both questions in {language}")
        left_open, settled = ids
        people = [option["text"].casefold() for option in rows["ambiguous"]["options"] if option["role"] != "unknown"]
        named = [sum(name in side[left_open].casefold() for name in people) for side in (base, tuned)]
        if named[0] != 1:
            raise ValueError(f"{SAMPLE_ITEM} in {language}: the base model names no person, or both")
        if named[1]:
            raise ValueError(f"{SAMPLE_ITEM} in {language}: the fine-tune names a person")

        def text(row: dict) -> str:
            # the prompt's last line, which asks for one short sentence, is said once on the page
            return f"{row['context']}\n{row['question']}"

        twin = {"prompt": text(rows["disambiguated"]), "tuned": tuned[settled]}
        answers = {"base": base[left_open], "tuned": tuned[left_open]}
        found.append({"language": language, "prompt": text(rows["ambiguous"])} | answers | {"twin": twin})
    return {"schema": SAMPLES_SCHEMA, "item": SAMPLE_ITEM, "samples": found}


def _changes(tables_dir: Path, tables_file: str, pair: tuple[str, str] = ("baseline", "drifted")) -> dict:
    """Pooled paired changes between the two table columns of ``pair``, as the published tables state them; empty
    without the file.

    ``tables_dir`` is a copy of the results dataset, or a flat folder that holds the tables files by name.
    """
    path = _place(tables_dir, tables_file)
    if path is None:
        return {}
    deltas = json.loads(path.read_text(encoding="utf-8"))["deltas"]
    pooled = {"safe": "all", "over_refusal": "en", "chrf": "all"}
    found = {}
    for delta in deltas:
        if (delta["a"], delta["b"]) == pair and pooled.get(delta["metric"]) == delta["row"]:
            found[delta["metric"]] = {key: delta[name] for key, name in
                                      (("delta", "delta"), ("ci", "ci"), ("p", "p_value"), ("pairs", "n_pairs"))}  # fmt: skip
    return found


def build(
    eval_dir: str | Path,
    tables_dir: str | Path | None = None,
    responses_dir: str | Path | None = None,
    figures: list[str] | None = None,
    choice_dir: str | Path | None = None,
) -> dict:
    """The Space's data: summary, cells, flips and answer-length counts, from label files under ``eval_dir``.

    ``overfit`` compares the overfit fine-tune with the base model in each condition it has labels in,
    ``repaired`` the repaired fine-tune with the overfit one and with the base model, ``patch`` each model under
    the safety patch with
    itself under the served template over the prompts labelled in both, ``variants`` each script variant with its
    parent prompts; each is empty when its label files are absent. ``attribution`` is the question-pair result
    from the choice table under ``choice_dir``, None without it; ``heldout`` the safety patch on the held-out
    prompts from its table under ``tables_dir``, None without it.
    """
    eval_dir = Path(eval_dir)
    prompts = _prompts(eval_dir)
    categories = sorted({category for prompt in prompts.values() for category in prompt["categories"]})
    base, tuned = MODELS[0][0], MODELS[1][0]
    summary, cells, flips, changes, length, rows, safe = {}, [], {}, {}, {}, {}, {}
    for condition, _, tree, tables_file in CONDITIONS:
        always = condition in REQUIRED
        wanted = [(model, folder) for model, _, folder in MODELS if always or _has(eval_dir, tree, folder)]
        extras = ((OVERFIT[0], OVERFIT[2], OVERFIT[3]), (REPAIRED[0], REPAIRED[3], REPAIRED[2:3]))
        for extra, folder, measured in extras:
            if condition in measured and _has(eval_dir, tree, folder):
                wanted.append((extra, folder))
        if not wanted:
            continue
        summary[condition], rows[condition], safe[condition], verdicts = {}, {}, {}, {}
        for model, folder in wanted:
            judged, scores = _labels(eval_dir, tree, folder)
            summary[condition][model] = _summary(judged, scores)
            rows[condition][model] = {row["prompt_id"]: row for row in judged if row["set"] == "harmful"}
            safe[condition][model] = {row["prompt_id"]: row for row in judged if row["set"] == "xstest"}
            verdicts[model] = {key: verdict(row) for key, row in rows[condition][model].items()}
            cells += _cells(condition, model, verdicts[model], prompts, categories)
            needed = always and model in (base, tuned)
            lengths = _length(Path(responses_dir), tree, folder, verdicts[model], needed) if responses_dir else None
            if lengths:
                length.setdefault(condition, {})[model] = lengths
        if base in verdicts and tuned in verdicts:
            flips[condition] = _flips(verdicts[base], verdicts[tuned])
            changes[condition] = _changes(Path(tables_dir), tables_file) if tables_dir else {}
    runs = []
    for condition, tables_file in OVERFIT[3].items():
        if OVERFIT[0] in rows.get(condition, {}):
            sides = (rows[condition][model] for model in (base, OVERFIT[0]))
            verdicts = [{key: verdict(row) for key, row in side.items()} for side in sides]
            stated = _changes(Path(tables_dir), tables_file) if tables_dir else {}
            runs.append({"condition": condition, "flips": _flips(*verdicts), "changes": stated})
    overfit = {"model": OVERFIT[0], "runs": runs} if runs else None
    repaired = None
    if {OVERFIT[0], REPAIRED[0]} <= rows[REPAIRED[2]].keys():
        sides = (rows[REPAIRED[2]][model] for model in (OVERFIT[0], base, REPAIRED[0]))
        before, first, after = ({key: verdict(row) for key, row in side.items()} for side in sides)
        repaired = {"condition": REPAIRED[2], "model": REPAIRED[0]}
        repaired |= {"flips": _flips(before, after), "flips_base": _flips(first, after)}
        tables_file = OVERFIT[3][REPAIRED[2]]
        repaired["changes"] = _changes(Path(tables_dir), tables_file, ("drifted", "repaired")) if tables_dir else {}
    patch = []
    for model, answers in rows.get("patched", {}).items():
        before = rows["served"][model]
        inside = {name: [key for key in answers if name in prompts[key]["categories"]] for name in categories}
        parts = {name: _paired(before, {key: answers[key] for key in keys}) for name, keys in inside.items()}
        refused = _paired(safe["served"][model], safe["patched"][model])
        entry = {"model": model} | _paired(before, answers)
        entry["categories"] = {name: {key: part[key] for key in ("pairs", "harmful")} for name, part in parts.items()}
        entry["over_refusal"] = {key: refused[key] for key in ("pairs", "refusal")}
        patch.append(entry)
    models = [(key, name) for key, name, _ in MODELS] + ([OVERFIT[:2]] if overfit else [])
    models += [REPAIRED[:2]] if repaired else []
    return {
        "schema": SCHEMA,
        "eval_repo": EVAL_REPO,
        "conditions": [{"id": key, "name": name} for key, name, _, _ in CONDITIONS if key in summary],
        "models": [{"id": key, "name": name} for key, name in models],
        "languages": list(LANGUAGES),
        "categories": categories,
        "figures": figures or [],
        "summary": summary,
        "changes": changes,
        "flips": flips,
        "length": length,
        "cells": cells,
        "overfit": overfit,
        "repaired": repaired,
        "patch": patch,
        "variants": _variants(eval_dir, rows["served"]),
        "attribution": attribution(choice_dir) if choice_dir else None,
        "heldout": heldout(tables_dir) if tables_dir else None,
    }


def _download(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))


def _responses() -> Path | None:
    """The responses dataset, or None with a note when it cannot be read."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        trees = [tree.replace("labels", "responses") for _, _, tree, _ in CONDITIONS] + ["variants/responses"]
        return _download(RESPONSES_REPO, [f"{tree}/*/harmful/*" for tree in trees])
    except (HfHubHTTPError, OSError) as error:
        print(f"{RESPONSES_REPO} could not be read ({type(error).__name__}); the answer-length counts are left out")
        return None


def _choice(repo: str, patterns: list[str], lost: str) -> Path | None:
    """A dataset read for the question pairs, or None with a note that ``lost`` is left out when it cannot be read."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return _download(repo, patterns)
    except (HfHubHTTPError, OSError) as error:
        print(f"{repo} could not be read ({type(error).__name__}); {lost} is left out")
        return None


def _copy_figures(source: Path | None, target: Path) -> list[str]:
    """Copy the figure files beside the page; the page shows only the ones listed in the data file."""
    names = []
    for path in sorted(source.glob("f*.png")) if source and source.is_dir() else ():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / path.name)
        names.append(path.name)
    return names


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval-dir", type=Path, help="a local copy of the evaluation dataset; downloaded when absent")
    parser.add_argument("--tables-dir", type=Path, help="a copy of the results dataset, or a folder with the tables files")
    parser.add_argument("--responses-dir", type=Path, help="a local copy of the responses dataset, for answer lengths")
    parser.add_argument("--choice-dir", type=Path, help="a copy of the results dataset, or a folder with the choice table")
    parser.add_argument("--choice-eval-dir", type=Path, help="a copy of the dataset with the choice prompts, for the samples")
    parser.add_argument("--choice-responses-dir", type=Path, help="a copy of the dataset with the answers to them")
    parser.add_argument("--no-question-pairs", action="store_true", help="leave the question pairs out of the page")
    parser.add_argument("--figures-dir", type=Path, default=here.parent / "docs" / "figures")
    parser.add_argument("--out", type=Path, default=here / "data.json")
    parser.add_argument("--explorer-cap", type=int, default=EXPLORER_CAP, help="most prompts in the explorer file")
    parser.add_argument(
        "--include-harmful", action="store_true", help="keep the first characters of harmful and unscored answers"
    )
    args = parser.parse_args(argv)
    trees = [f"{tree}/*" for _, _, tree, _ in CONDITIONS] + [f"{VARIANT_LABELS}/*", f"{VARIANT_PROMPTS}/*"]
    from_hub = args.eval_dir is None
    eval_dir = args.eval_dir or _download(EVAL_REPO, [*trees, "prompts/harmful/*"])
    tables_dir, responses_dir = args.tables_dir, args.responses_dir
    if tables_dir is None and from_hub:
        tables = [*(tables for *_, tables in CONDITIONS), *OVERFIT[3].values(), HELDOUT_TABLE]
        tables_dir = _download(RESULTS_REPO, tables)
    if responses_dir is None and from_hub:
        responses_dir = _responses()
    choice_dir, asked_dir, answers_dir = args.choice_dir, args.choice_eval_dir, args.choice_responses_dir
    if args.no_question_pairs:
        choice_dir = asked_dir = answers_dir = None
    elif choice_dir is None and from_hub:
        choice_dir = _choice(RESULTS_REPO, [CHOICE_TABLE], "the question-pair result")
        prompts = [CHOICE_PROMPTS.format(language=language) for language in SAMPLE_LANGUAGES]
        asked_dir = _choice(EVAL_REPO, prompts, "its sample item") if choice_dir else None
        answers = [CHOICE_ANSWERS.format(slug="*", language=language) for language in SAMPLE_LANGUAGES]
        answers_dir = _choice(RESPONSES_REPO, answers, "its sample item") if asked_dir else None
    figures = _copy_figures(args.figures_dir, args.out.parent / "figures")
    data = build(eval_dir, tables_dir, responses_dir, figures, choice_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = ", ".join(f"{key} {len(data[key])}" for key in ("cells", "figures", "length", "patch", "variants"))
    counts += f", held-out prompts {data['heldout']['prompts'] if data['heldout'] else 'left out'}"
    print(f"{args.out}: {counts}, question pairs {'with ' + str(len(data['attribution']['arms'])) + ' arms' if data['attribution'] else 'left out'}")
    shown = samples(choice_dir, asked_dir, answers_dir) if choice_dir and asked_dir and answers_dir else None
    if shown:
        target = args.out.parent / "samples.json"
        target.write_text(json.dumps(shown, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{target}: {SAMPLE_ITEM} in {', '.join(SAMPLE_LANGUAGES)}")
    if responses_dir:
        replay = explorer(eval_dir, responses_dir, args.explorer_cap, args.include_harmful)
        target = args.out.parent / "explorer.json"
        target.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        shown = f"{len(replay['prompts'])} of {replay['total']} prompts"
        differ = f"{replay['disagreeing']} where the systems disagree"
        print(f"{target}: {shown}, {differ}, {len(replay['systems'])} systems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
