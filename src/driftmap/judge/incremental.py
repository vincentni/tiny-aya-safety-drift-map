"""Label files that grow one batch of judged rows at a time."""

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from driftmap.core.records import GenerationRow, JudgeConfig, LabelRow, Record
from driftmap.core.schemas import SCHEMA_LABEL, DriftmapError, ExitCode
from driftmap.judge import parser
from driftmap.judge.parser import ParsedLabels
from driftmap.table.build import DuplicateKey

UNSCORED = "unscored"
FIELDS = ("request_harm", "refusal", "response_harm")
# a judge answer is 100 tokens at most
MAX_RAW = 2000

Key = tuple[str, str, str]


class RawRow(Record):
    """What the judge wrote for one row, kept so a parser fix needs no model."""

    checkpoint: str
    language: str
    prompt_id: str
    repo: str
    revision: str
    response_sha256: str
    raw: str


def key(row: GenerationRow | LabelRow | RawRow) -> Key:
    return row.checkpoint, row.language, row.prompt_id


def response_sha256(response: str) -> str:
    return hashlib.sha256(response.encode()).hexdigest()


def read_rows[T: Record](path: str | Path, model: type[T], name: str | None = None) -> list[T]:
    """Rows of a JSONL file, none when it is absent; DriftmapError naming ``name`` and the line on a bad row."""
    path = Path(path)
    name = name or path.name
    if not path.is_file():
        return []
    rows = []
    # splitlines() would also cut at U+2028 inside a prompt
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValidationError as error:
            fault = error.errors()[0]
            found = f"{'.'.join(map(str, fault['loc']))}: {fault['msg']}"
            # the pydantic error carries input values, so it is not chained
            raise DriftmapError(ExitCode.INTEGRITY, f"{name}:{number}: not a {model.__name__} ({found})") from None
    return rows


def dump_rows(rows: Iterable[Record]) -> bytes:
    return "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()


def check_unique(rows: Iterable[GenerationRow], name: str) -> None:
    """Raises DuplicateKey naming ``name`` when two rows share a prompt id."""
    counts = Counter(row.prompt_id for row in rows)
    repeated = sorted(prompt_id for prompt_id, count in counts.items() if count > 1)
    if repeated:
        raise DuplicateKey(f"{name} has prompt {repeated[0]} more than once")


def _stands(label: LabelRow, raw: RawRow | None, row: GenerationRow) -> bool:
    """True while the stored judge output of ``row``, read now, gives exactly this label."""
    if label.rendered_prefix_sha256 != row.rendered_prefix_sha256:
        # the label would keep the prefix hash of an earlier rendering
        return False
    if row.response is None or not label.raw_output_present:
        # a row never sent to the judge has no stored output
        return row.response is None and not label.raw_output_present
    if raw is None or (raw.repo, raw.revision) != (label.judge.repo, label.judge.revision):
        return False
    if raw.response_sha256 != response_sha256(row.response):
        return False
    read = parser.parse(raw.raw) or ParsedLabels(UNSCORED, UNSCORED, UNSCORED)
    return read == (label.request_harm, label.refusal, label.response_harm)


def pending(
    responses: Iterable[GenerationRow],
    labels: Iterable[LabelRow],
    config_hash: str,
    rejudge_unscored: bool = False,
    *,
    raws: Iterable[RawRow],
) -> list[GenerationRow]:
    """Rows with no standing label under ``config_hash``; with ``rejudge_unscored`` also answered unscored rows.

    A label stands while the stored output in ``raws`` is of the label's judge, answers the row's current response
    and parses to the label's three values; the label file alone is never trusted.
    """
    labelled = {key(row): row for row in labels if row.judge is not None and row.judge.config_hash == config_hash}
    stored = {key(raw): raw for raw in raws}
    waiting = []
    for row in responses:
        label = labelled.get(key(row))
        stands = label is not None and _stands(label, stored.get(key(row)), row)
        if not stands or (rejudge_unscored and row.response is not None and label.response_harm == UNSCORED):
            waiting.append(row)
    return waiting


def union[T: LabelRow | RawRow](existing: Iterable[T], new: Iterable[T]) -> list[T]:
    """One row per (checkpoint, language, prompt id), the new one winning, in key order."""
    merged = {key(row): row for row in existing} | {key(row): row for row in new}
    return [merged[name] for name in sorted(merged)]


def label_row(row: GenerationRow, config: JudgeConfig, labels: ParsedLabels | None, raw_present: bool) -> LabelRow:
    """The label row of a generation row; without labels all three read unscored."""
    values = labels or ParsedLabels(UNSCORED, UNSCORED, UNSCORED)
    return LabelRow(
        schema=SCHEMA_LABEL,
        checkpoint=row.checkpoint,
        language=row.language,
        set=row.set,
        prompt_id=row.prompt_id,
        prompt=row.prompt,
        rendered_prefix_sha256=row.rendered_prefix_sha256,
        **values._asdict(),
        raw_output_present=raw_present,
        judge=config,
        chrf=None,
    )


def judged_rows(
    rows: Sequence[GenerationRow], judged: Sequence[tuple[str, ParsedLabels | None]], config: JudgeConfig
) -> tuple[list[LabelRow], list[RawRow]]:
    """Label and raw rows for ``rows``; ``judged`` holds one answer per row that has a response, in order."""
    answered = sum(row.response is not None for row in rows)
    if answered != len(judged):
        raise DriftmapError(ExitCode.INTEGRITY, f"the judge gave {len(judged)} answers for {answered} responses")
    answers = iter(judged)
    labels, raws = [], []
    for row in rows:
        if row.response is None:
            # nothing to judge: the generation itself is unscored
            labels.append(label_row(row, config, None, False))
            continue
        raw, parsed = next(answers)
        labels.append(label_row(row, config, parsed, True))
        raws.append(
            RawRow(
                checkpoint=row.checkpoint,
                language=row.language,
                prompt_id=row.prompt_id,
                repo=config.repo,
                revision=config.revision,
                response_sha256=response_sha256(row.response),
                raw=raw[:MAX_RAW],
            )
        )
    return labels, raws


def counts(labels: Iterable[LabelRow]) -> dict[str, dict[str, int]]:
    """Per label, how many judged rows read yes, no and unscored."""
    judged = [row for row in labels if row.judge is not None]
    return {
        field: {value: sum(getattr(row, field) == value for row in judged) for value in ("yes", "no", UNSCORED)}
        for field in FIELDS
    }
