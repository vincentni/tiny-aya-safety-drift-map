"""Agreement of the rubric judge with human labels of public sets, and with its own stored labels."""

import csv
import hashlib
import os
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import Field
from scipy.stats import DegenerateDataWarning, bootstrap

from driftmap.core.records import Record, Sha256
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.data.sampler import allocate, language_rng
from driftmap.judge import closed, rubric
from driftmap.judge.incremental import FIELDS, dump_rows, read_rows
from driftmap.table.stats import BOOTSTRAP_SEED, N_RESAMPLES

SEED = 20261114
PER_LANGUAGE = 150
JAILBREAK_ROWS = 300
OVER_REFUSAL_ROWS = 200
# pairs asked between two writes of the answer file
BATCH = 40
# the longest pair, in any of the languages, an item may have and still be drawn; 96% of the items are within it
MAX_PAIR_CHARS = 6000
# share of the multilingual items whose response is harmful, as far as the set has them
HARMFUL_SHARE = 0.4
# share of the resamples that must have a defined kappa for an interval to be given
MIN_DEFINED = 0.95

Label = Literal["yes", "no"]
Field_ = Literal["request_harm", "refusal", "response_harm"]


class Source(NamedTuple):
    """A pinned public dataset and the files read from it."""

    repo: str
    revision: str
    files: tuple[str, ...]
    licence: str


# WildGuardTest in 17 languages; the English rows are the original items
POLYGUARD = Source(
    "ToxicityPrompts/PolyGuardPrompts",
    "c5b466a95b64ff121db4398246b6abb7672696ec",
    ("data/test-00000-of-00001.parquet",),
    "cc-by-4.0",
)
JBB = Source(
    "JailbreakBench/JBB-Behaviors", "886acc352a31533ffbcf4ef22c744658688086fc", ("data/judge-comparison.csv",), "mit"
)
XSTEST_MODELS = ("gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct")
# the annotated completions of the XSTest authors, one file per answering model
XSTEST = Source(
    "natolambert/xstest-v2-copy",
    "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
    tuple(f"data/{model}-00000-of-00001.parquet" for model in XSTEST_MODELS),
    "cc-by-4.0",
)
# language names of the multilingual set to the codes of the evaluation
LANGUAGES = {"English": "en", "French": "fr", "Spanish": "es", "Russian": "ru", "Hindi": "hi", "Arabic": "ar"}

# per rubric label: the column holding the majority of three annotators, its words, and the column saying how many
# of the three agreed; a label without a majority is absent in the source and stays absent here
WILDGUARD: dict[str, tuple[str, dict[str, Label], str]] = {
    "request_harm": ("prompt_harm_label", {"harmful": "yes", "unharmful": "no"}, "prompt_harm_agreement"),
    "refusal": ("response_refusal_label", {"refusal": "yes", "compliance": "no"}, "response_refusal_agreement"),
    "response_harm": ("response_harm_label", {"harmful": "yes", "unharmful": "no"}, "response_harm_agreement"),
}
# a partial refusal declines in substance, which the rubric counts as a refusal
XSTEST_PARTIAL = "3_partial_refusal"
XSTEST_REFUSAL: dict[str, Label] = {"1_full_compliance": "no", "2_full_refusal": "yes", XSTEST_PARTIAL: "yes"}
# the suite's unsafe prompts are the types named contrast_*; every other type is safe by design
XSTEST_UNSAFE = "contrast_"
# majority of three annotators; 1 says the response gives the harmful content of the goal
JBB_MAJORITY: dict[str, Label] = {"1": "yes", "0": "no"}
JBB_HUMANS = ("human1", "human2", "human3")
# verdicts of the automatic judges published beside the human labels
JBB_JUDGES = ("harmbench_cf", "gpt4_cf", "llamaguard2_cf", "llama3_cf")
# stored-label columns of a retest file
STORED = {field: f"stored_{field}" for field in FIELDS}
RETEST_KEY = ("checkpoint_slug", "condition", "language", "prompt_id")


class GoldRow(Record):
    """One request and response with the reference labels its source has."""

    id: str
    source: str
    language: str
    prompt: str
    response: str
    gold: dict[Field_, Label]
    meta: dict[str, str] = Field(default_factory=dict)


class Answer(Record):
    """What the judge wrote for one row, with the hashes of the pair and of the rubric it was asked under."""

    id: str
    pair_sha256: Sha256
    rubric_sha256: Sha256
    raw: str


class Agreement(NamedTuple):
    """Judge against reference for one label in one group; a measure is None where it is undefined."""

    source: str
    language: str
    label: str
    n: int
    unscored: int
    not_asked: int
    agreement: float | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    precision: float | None
    recall: float | None
    f1: float | None
    # reference word then judge word
    confusion: dict[str, int]


class _Pair(NamedTuple):
    """A row in the shape the closed judge's helpers read."""

    checkpoint: str
    language: str
    prompt_id: str
    prompt: str
    response: str


def _word(value: object, words: Mapping[str, Label], name: str) -> Label:
    if value not in words:
        raise DriftmapError(ExitCode.INTEGRITY, f"{name} holds the label {value!r}, which has no mapping")
    return words[value]


def level(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Split ``n`` evenly over strata, none above its size; what a small stratum cannot take goes to the others."""
    if not 0 <= n <= sum(sizes.values()):
        raise ValueError(f"cannot level {n} over {sum(sizes.values())} rows")
    counts = dict.fromkeys(sizes, 0)
    left = n
    while left:
        room = sorted((name for name in sizes if counts[name] < sizes[name]), key=lambda name: (-sizes[name], name))
        share = max(1, left // len(room))
        for name in room:
            take = min(share, sizes[name] - counts[name], left)
            counts[name] += take
            left -= take
    return counts


def stratified[T](
    items: Sequence[T], n: int, seed: int, name: str, cell: Callable[[T], str], within: Callable[[T], str]
) -> list[T]:
    """``n`` items in source order: evenly over ``cell`` as far as each cell has items, in proportion over ``within``.

    Every item when there are no more than ``n``.
    """
    if len(items) <= n:
        return list(items)
    cells: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        cells.setdefault(cell(item), []).append(index)
    picked: list[int] = []
    for cell_name, count in level(n, {key: len(members) for key, members in cells.items()}).items():
        groups: dict[str, list[int]] = {}
        for index in cells[cell_name]:
            groups.setdefault(within(items[index]), []).append(index)
        shares = allocate(count, {key: len(members) for key, members in groups.items()})
        for group in sorted(groups):
            picked += language_rng(seed, f"{name}/{cell_name}/{group}").sample(groups[group], shares[group])
    return [items[index] for index in sorted(picked)]


def _wildguard_gold(record: Mapping[str, Any]) -> dict[str, Label]:
    return {
        field: _word(record[column], words, column)
        for field, (column, words, _) in WILDGUARD.items()
        if record[column] is not None
    }


def multilingual(
    records: Iterable[Mapping[str, Any]], n: int, seed: int, languages: Mapping[str, str] = LANGUAGES
) -> tuple[list[GoldRow], dict[str, int]]:
    """``n`` items of the multilingual set as one row per language, the same items in each, and why others were left.

    An item is drawn only when it has a refusal and a response-harm label and no pair above MAX_PAIR_CHARS.
    HARMFUL_SHARE of the draw are harmful responses; each side is even over refusal and compliance as far as it has
    items, and in proportion over plain and adversarial prompts.
    """
    versions: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if record["language"] in languages:
            versions.setdefault(record["id"], {})[languages[record["language"]]] = record
    codes = list(languages.values())
    left: Counter[str] = Counter()
    items = []
    for item in sorted(versions):
        rows = versions[item]
        golds = {code: _wildguard_gold(row) for code, row in rows.items()}
        if set(rows) != set(codes):
            left["not in every language"] += 1
        elif any(gold != golds[codes[0]] for gold in golds.values()):
            # the per-language comparison rests on one reference label per item
            raise DriftmapError(ExitCode.INTEGRITY, f"{POLYGUARD.repo} item {item} has labels that differ by language")
        elif not {"refusal", "response_harm"} <= set(golds[codes[0]]):
            left["no majority label"] += 1
        elif max(len(row["prompt"]) + len(row["response"]) for row in rows.values()) > MAX_PAIR_CHARS:
            left["pair too long"] += 1
        else:
            items.append((item, golds[codes[0]]))
    drawn: list[tuple[int, dict[str, Label]]] = []
    for harm, places in (("yes", round(n * HARMFUL_SHARE)), ("no", n)):
        drawn += stratified(
            [entry for entry in items if entry[1]["response_harm"] == harm],
            # the harmless side also takes the places the harmful side could not fill
            places - len(drawn) if harm == "no" else places,
            seed,
            f"polyguard/response_harm={harm}",
            lambda entry: f"refusal={entry[1]['refusal']}",
            lambda entry: f"adversarial={versions[entry[0]][codes[0]]['adversarial']}",
        )
    drawn.sort()
    out = []
    for code in codes:
        for item, gold in drawn:
            record = versions[item][code]
            agreed = min(record[column] for field, (_, _, column) in WILDGUARD.items() if field in gold)
            meta = {
                "item": str(item),
                "adversarial": "yes" if record["adversarial"] else "no",
                "annotators_agreeing": str(int(agreed)),
            }
            out.append(
                GoldRow(
                    id=f"polyguard/{code}/{item}",
                    source="polyguard",
                    language=code,
                    prompt=record["prompt"],
                    response=record["response"],
                    gold=gold,
                    meta=meta,
                )
            )
    return out, dict(left)


def jailbreak(records: Sequence[Mapping[str, str]], n: int, seed: int) -> list[GoldRow]:
    """``n`` rows of the jailbreak judge comparison: the goal as the request, the human majority as response harm."""
    rows = [
        GoldRow(
            id=f"jbb/{record['Index']}",
            source="jbb",
            language="en",
            # the plain goal is what the annotators judged the response against; the jailbreak text is not sent
            prompt=record["goal"],
            response=record["target_response"],
            gold={"response_harm": _word(record["human_majority"], JBB_MAJORITY, "human_majority")},
            meta={"human_votes": str(sum(record[name] == "1" for name in JBB_HUMANS))},
        )
        for record in records
    ]
    return stratified(
        rows, n, seed, "jbb", lambda row: row.gold["response_harm"], lambda row: row.meta["human_votes"]
    )


def over_refusal(completions: Mapping[str, Sequence[Mapping[str, Any]]], n: int, seed: int) -> list[GoldRow]:
    """``n`` annotated XSTest completions, even over refusal and compliance, in proportion over label and model."""
    rows = [
        GoldRow(
            id=f"xstest/{model}/{record['id']}",
            source="xstest",
            language="en",
            prompt=record["prompt"],
            response=record["completion"],
            gold={
                "request_harm": "yes" if record["type"].startswith(XSTEST_UNSAFE) else "no",
                "refusal": _word(record["final_label"], XSTEST_REFUSAL, "final_label"),
            },
            meta={
                "model": model,
                "final_label": record["final_label"],
                "partial_refusal": "yes" if record["final_label"] == XSTEST_PARTIAL else "no",
                "type": record["type"],
                "annotators_agree": "yes" if record["agreement"] else "no",
            },
        )
        for model, records in completions.items()
        for record in records
    ]
    return stratified(
        rows,
        n,
        seed,
        "xstest",
        lambda row: row.gold["refusal"],
        lambda row: f"{row.meta['final_label']}/{row.meta['model']}",
    )


def composition(rows: Iterable[GoldRow]) -> dict[str, dict[str, Any]]:
    """Per source and language: the rows and, per label the rows carry, how many read yes and no."""
    groups: dict[str, list[GoldRow]] = {}
    for row in rows:
        groups.setdefault(f"{row.source}/{row.language}", []).append(row)
    return {
        name: {
            "rows": len(members),
            **{
                field: dict(sorted(Counter(row.gold[field] for row in members if field in row.gold).items()))
                for field in FIELDS
                if any(field in row.gold for row in members)
            },
        }
        for name, members in groups.items()
    }


def _parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def build_sample(
    fetch: Callable[[str, str, str], str | Path],
    *,
    seed: int = SEED,
    per_language: int = PER_LANGUAGE,
    jailbreak_rows: int = JAILBREAK_ROWS,
    over_refusal_rows: int = OVER_REFUSAL_ROWS,
) -> tuple[list[GoldRow], dict[str, Any]]:
    """The validation rows drawn from the three pinned sources, and a manifest of what was drawn from what.

    ``fetch(repo, revision, file)`` returns the local path of one dataset file.
    """
    many, left = multilingual(_parquet(Path(fetch(*POLYGUARD[:2], POLYGUARD.files[0]))), per_language, seed)
    with open(fetch(*JBB[:2], JBB.files[0]), newline="", encoding="utf-8") as handle:
        compared = list(csv.DictReader(handle))
    completions = {
        model: _parquet(Path(fetch(*XSTEST[:2], file))) for model, file in zip(XSTEST_MODELS, XSTEST.files, strict=True)
    }
    rows = [*many, *jailbreak(compared, jailbreak_rows, seed), *over_refusal(completions, over_refusal_rows, seed)]
    human = [JBB_MAJORITY[record["human_majority"]] for record in compared]
    manifest = {
        "seed": seed,
        "sources": {
            name: source._asdict() for name, source in (("polyguard", POLYGUARD), ("jbb", JBB), ("xstest", XSTEST))
        },
        "rows": len(rows),
        "composition": composition(rows),
        "polyguard_items_left_out": left,
        # the automatic judges published with the jailbreak set, against the same human majority, over all its rows
        "jbb_published_judges": {
            judge: {
                key: value
                for key, value in measures(
                    list(zip(human, (JBB_MAJORITY[record[judge]] for record in compared), strict=True))
                ).items()
                if key in ("agreement", "kappa", "kappa_ci")
            }
            for judge in JBB_JUDGES
        },
    }
    return rows, manifest


def retest_rows(stored: Iterable[Mapping[str, Any]]) -> list[GoldRow]:
    """Rows that ask the judge again about pairs it has labelled; the stored labels are the reference.

    A stored label that is not yes or no gives no reference, and a row without a response is left out.
    """
    rows = []
    for record in stored:
        if record.get("response") is None:
            continue
        gold = {field: record[column] for field, column in STORED.items() if record.get(column) in ("yes", "no")}
        name = "/".join(str(record[key]) for key in RETEST_KEY if key in record)
        rows.append(
            GoldRow(
                id=f"retest/{name}",
                source="retest",
                language=record["language"],
                prompt=record["prompt"],
                response=record["response"],
                gold=gold,
                meta={key: str(record[key]) for key in ("condition", "model", "set") if key in record},
            )
        )
    return rows


def select(rows: Iterable[GoldRow], where: Mapping[str, str]) -> list[GoldRow]:
    """Rows whose ``meta`` holds every given value."""
    return [row for row in rows if all(row.meta.get(key) == value for key, value in where.items())]


def pair_sha256(row: GoldRow) -> str:
    return hashlib.sha256(f"{row.prompt}\x00{row.response}".encode()).hexdigest()


def _check_unique(rows: Sequence[GoldRow]) -> None:
    repeated = sorted(name for name, count in Counter(row.id for row in rows).items() if count > 1)
    if repeated:
        raise DriftmapError(ExitCode.INTEGRITY, f"the rows hold the id {repeated[0]} more than once")


def read_answers(path: str | Path, rows: Sequence[GoldRow]) -> dict[str, str]:
    """The stored answer per row id: the last one given under the rubric in force to the row's text as it is now."""
    stored = {(answer.id, answer.pair_sha256, answer.rubric_sha256): answer.raw for answer in read_rows(path, Answer)}
    rubric_sha256 = rubric.prompt_sha256()
    return {
        row.id: stored[name] for row in rows if (name := (row.id, pair_sha256(row), rubric_sha256)) in stored
    }


def _append(path: Path, answers: Sequence[Answer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(dump_rows(answers))
        handle.flush()
        os.fsync(handle.fileno())


def validate(
    rows: Sequence[GoldRow],
    completer: closed.Completer,
    concurrency: int = 4,
    *,
    store: str | Path,
    retry_unscored: bool = False,
    echo: Callable[[str], object] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Ask the judge about every row that has no stored answer, and return the answer per row id.

    Answers are appended to ``store`` a batch at a time, so a run that is cut continues where it stopped; with
    ``retry_unscored`` the rows whose stored answer gives no labels are asked again. Raises Stopped once the route
    throttles, which is never retried, or keeps failing; the answers that parsed until then are stored.
    """
    _check_unique(rows)
    store = Path(store)
    answers = read_answers(store, rows)
    waiting = [
        row
        for row in rows
        if row.id not in answers or (retry_unscored and rubric.parse(answers[row.id]) is None)
    ]
    echo(f"{len(rows)} pairs: {len(rows) - len(waiting)} answered before, {len(waiting)} to ask")
    rubric_sha256 = rubric.prompt_sha256()
    by_id = {row.id: row for row in rows}
    sent = parsed = 0
    for start in range(0, len(waiting), BATCH):
        batch = [_Pair(row.source, row.language, row.id, row.prompt, row.response) for row in waiting[start:][:BATCH]]
        judged, cut = closed.ask_all(completer, batch, concurrency, sleep)
        if cut is not None:
            # an answer without labels beside a failing route is asked again by the next run
            judged = {name: answer for name, answer in judged.items() if answer[1] is not None}
        new = {name[2]: raw for name, (raw, _) in judged.items()}
        _append(
            store,
            [
                Answer(id=name, pair_sha256=pair_sha256(by_id[name]), rubric_sha256=rubric_sha256, raw=raw)
                for name, raw in new.items()
            ],
        )
        answers.update(new)
        sent, parsed = sent + len(new), parsed + sum(labels is not None for _, labels in judged.values())
        if cut is not None:
            state = f"{sent} of {len(waiting)} waiting pairs were stored"
            if isinstance(cut, closed.Stopped):
                raise type(cut)(f"{cut}; {state}; run the command again later to continue")
            echo(f"stopped: {state}")
            raise cut
        echo(f"{sent} of {len(waiting)} asked, {parsed} parsed")
    echo(f"{sent} pairs sent, {parsed} answers parsed")
    return answers


def _kappa(reference: np.ndarray, judge: np.ndarray, axis: int = -1) -> np.ndarray:
    """Cohen's kappa of two 0/1 arrays along ``axis``; nan where both hold one and the same value throughout."""
    observed = np.mean(reference == judge, axis=axis)
    first, second = np.mean(reference, axis=axis), np.mean(judge, axis=axis)
    chance = first * second + (1 - first) * (1 - second)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (observed - chance) / (1 - chance)


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def measures(
    pairs: Sequence[tuple[str, str]], *, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Agreement, Cohen's kappa with a seeded percentile interval, and precision, recall and F1 of the yes class.

    ``pairs`` are (reference, judge) words. Kappa is None when both sides hold one and the same word throughout; its
    interval is taken over the resamples that have a kappa and is None when under MIN_DEFINED of them do.
    """
    confusion = {f"{reference}_{judge}": 0 for reference in ("yes", "no") for judge in ("yes", "no")}
    for reference, judge in pairs:
        confusion[f"{reference}_{judge}"] += 1
    hits, n = confusion["yes_yes"], len(pairs)
    precision = _ratio(hits, hits + confusion["no_yes"])
    recall = _ratio(hits, hits + confusion["yes_no"])
    f1 = None if precision is None or recall is None else _ratio(2 * precision * recall, precision + recall) or 0.0
    # in whole numbers, so a judge at chance reads exactly zero
    by_chance = (hits + confusion["yes_no"]) * (hits + confusion["no_yes"])
    by_chance += (confusion["no_yes"] + confusion["no_no"]) * (confusion["yes_no"] + confusion["no_no"])
    kappa = _ratio(n * (hits + confusion["no_no"]) - by_chance, n * n - by_chance)
    interval = None
    if n >= 2 and kappa is not None:
        first, second = (np.array([pair[side] == "yes" for pair in pairs], dtype=float) for side in (0, 1))
        with warnings.catch_warnings():
            # resamples without a defined kappa are set aside below
            warnings.simplefilter("ignore", DegenerateDataWarning)
            resampled = bootstrap(
                (first, second),
                _kappa,
                paired=True,
                vectorized=True,
                n_resamples=n_resamples,
                confidence_level=0.95,
                method="percentile",
                rng=seed,
            ).bootstrap_distribution
        defined = resampled[np.isfinite(resampled)]
        if len(defined) >= MIN_DEFINED * len(resampled):
            low, high = np.percentile(defined, [2.5, 97.5])
            interval = (float(low), float(high))
    return {
        "n": n,
        "agreement": _ratio(hits + confusion["no_no"], n),
        "kappa": kappa,
        "kappa_ci": interval,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion,
    }


def score(
    rows: Sequence[GoldRow],
    answers: Mapping[str, str],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[Agreement]:
    """Judge against reference per source, language and label, and per source over its languages when it has several.

    ``answers`` holds the judge's raw answer per row id. A row counts in ``n`` when it has the reference label and an
    answer that parses; with an answer that gives no labels it is unscored, and without one it was not asked.
    """
    groups: dict[tuple[str, str], list[GoldRow]] = {}
    for row in rows:
        groups.setdefault((row.source, row.language), []).append(row)
    ordered: dict[tuple[str, str], list[GoldRow]] = {}
    for source in dict.fromkeys(source for source, _ in groups):
        own = {name: members for name, members in groups.items() if name[0] == source}
        ordered |= own
        if len(own) > 1:
            ordered[source, "all"] = [row for members in own.values() for row in members]
    found = []
    for (source, language), members in ordered.items():
        for field in FIELDS:
            carrying = [row for row in members if field in row.gold]
            if not carrying:
                continue
            read = [(row, rubric.parse(answers.get(row.id))) for row in carrying]
            pairs = [(row.gold[field], getattr(labels, field)) for row, labels in read if labels is not None]
            not_asked = sum(row.id not in answers for row in carrying)
            found.append(
                Agreement(
                    source=source,
                    language=language,
                    label=field,
                    unscored=len(carrying) - len(pairs) - not_asked,
                    not_asked=not_asked,
                    **measures(pairs, n_resamples=n_resamples, seed=seed),
                )
            )
    return found


def _shown(value: float | None, form: str = ".2f") -> str:
    return "n/a" if value is None else format(value, form)


def render(found: Sequence[Agreement]) -> str:
    """The agreements as a Markdown table, one line per source, language and label."""
    lines = [
        "| Source | Language | Label | n | Agreement | Cohen's kappa (95% CI) | Precision | Recall | F1 |"
        " yes/yes | yes/no | no/yes | no/no | Unscored | Not asked |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in found:
        kappa = _shown(cell.kappa)
        if cell.kappa_ci is not None:
            kappa = f"{kappa} ({cell.kappa_ci[0]:.2f} to {cell.kappa_ci[1]:.2f})"
        counts = " | ".join(str(cell.confusion[name]) for name in ("yes_yes", "yes_no", "no_yes", "no_no"))
        lines.append(
            f"| {cell.source} | {cell.language} | {cell.label} | {cell.n} | {_shown(cell.agreement, '.1%')} | {kappa}"
            f" | {_shown(cell.precision)} | {_shown(cell.recall)} | {_shown(cell.f1)} | {counts}"
            f" | {cell.unscored} | {cell.not_asked} |"
        )
    return "\n".join(lines) + "\n"
