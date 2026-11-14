"""A Hub holding one checkpoint's generation rows and the repos a judge run writes to; placeholder text only."""

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from driftmap.core import ids
from driftmap.core.ids import judge_raw_path, labels_path, responses_path
from driftmap.core.records import Decoding, GenerationRow, JudgeConfig, LabelRow
from driftmap.core.schemas import SCHEMA_GENERATION
from driftmap.judge import rubric
from driftmap.judge.incremental import RawRow, judged_rows, read_rows
from tests.fakes import FakeHub
from tests.fakes.settings import BASE, settings_toml

# answers as the rubric asks for them
CANONICAL = '{"request_harm": "yes", "refusal": "no", "response_harm": "yes"}'
SAFE = '{"request_harm": "no", "refusal": "no", "response_harm": "no"}'
# answers the strict reader gives no labels for
UNPARSEABLE = [
    "The response is harmful.",
    '```json\n{"request_harm": "yes", "refusal": "no", "response_harm": "yes"}\n```',
    '{"request_harm": "yes", "refusal": "N/A", "response_harm": "yes"}',
]

OWNER = BASE["owner"]
RESPONSES = f"{OWNER}/{BASE['repos']['responses']}"
EVAL = f"{OWNER}/{BASE['repos']['eval']}"
MODEL = BASE["sources"]["model"]
CHECKPOINT = f"{MODEL}@{'1' * 40}"
SLUG = f"tiny-aya-global--{'1' * 12}"
PREFIX = "f" * 64
LANGUAGES = tuple(BASE["languages"])


def generation_rows(set_: str, language: str, n: int, *, checkpoint: str = CHECKPOINT) -> list[GenerationRow]:
    return [
        GenerationRow(
            schema=SCHEMA_GENERATION,
            checkpoint=checkpoint,
            language=language,
            set=set_,
            prompt_id=f"{set_}-{language}-{index:03d}",
            prompt=f"placeholder {set_} prompt {language} {index:03d}",
            rendered_prompt=f"<rendered> placeholder {set_} prompt {language} {index:03d}",
            rendered_prefix_sha256=PREFIX,
            response=f"placeholder {set_} response {language} {index:03d}",
            decoding=Decoding(runtime="transformers"),
            status="ok",
            error=None,
            prompt_tokens=12,
            generated_tokens=8,
            stop_reason="eos",
        )
        for index in range(n)
    ]


def jsonl(rows: Iterable[Any]) -> bytes:
    return "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()


def other_judge(config: JudgeConfig, name: str = "closed-model-b") -> JudgeConfig:
    """The config of another model under the same rubric."""
    other = config.model_copy(update={"repo": name})
    return other.model_copy(update={"config_hash": rubric.config_hash(other)})


class JudgeWorld:
    """A Hub with one checkpoint's stored responses and the evaluation dataset the labels go to."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch,
        *,
        languages=LANGUAGES,
        n=4,
        benign=False,
        mode="full",
        checkpoint=CHECKPOINT,
        **overrides,
    ) -> None:
        (tmp_path / "scratch").mkdir(parents=True)
        # the run's working directories go with the test's own
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        self.mode, self.checkpoint, self.slug = mode, ids.parse(checkpoint), ids.slug(checkpoint)
        self.fake = FakeHub(tmp_path / "hub")
        self.rows = {
            (set_, language): generation_rows(set_, language, n, checkpoint=checkpoint)
            for set_, codes in (("harmful", languages), ("xstest", ("en",)), ("benign", languages if benign else ()))
            for language in codes
        }
        files = {responses_path(self.slug, set_, code, mode): jsonl(rows) for (set_, code), rows in self.rows.items()}
        for repo in (RESPONSES, EVAL):
            self.fake.add_repo(repo, repo_type="dataset", private=True, files=files if repo == RESPONSES else {})
            self.fake.authorize(repo)
        self.config = settings_toml(tmp_path, languages=list(languages), **overrides)

    def put(self, path: str, data: bytes, repo: str) -> None:
        self.fake.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")

    def drop(self, path: str, repo: str) -> None:
        """Remove a file from the repo's head, as a commit would."""
        for revision in ("main", self.fake.heads[repo]):
            del self.fake.repos[repo][revision][path]

    def put_rows(self, set_: str, language: str, rows: Iterable[GenerationRow]) -> None:
        """Replace one responses file, as a later upload of the generation run."""
        self.put(responses_path(self.slug, set_, language, self.mode), jsonl(rows), RESPONSES)

    def seed(self, config: JudgeConfig, raw: str = SAFE, special: Mapping[str, str] | None = None) -> None:
        """Store each judged unit's labels and answers as a run under ``config`` leaves them."""
        files: dict[str, tuple[list[LabelRow], list[RawRow]]] = {}
        for (set_, language), rows in self.rows.items():
            if set_ == "benign":
                continue
            answers = [(special or {}).get(row.prompt_id, raw) for row in rows]
            labels, raws = judged_rows(rows, [(answer, rubric.parse(answer)) for answer in answers], config)
            kept = files.setdefault(language, ([], []))
            kept[0].extend(labels)
            kept[1].extend(raws)
        for language, (labels, raws) in files.items():
            self.put(judge_raw_path(self.slug, language, self.mode), jsonl(raws), RESPONSES)
            self.put(labels_path(self.slug, language, self.mode), jsonl(labels), EVAL)

    def stored(self, path: str, repo: str) -> bytes | None:
        return self.fake.repos[repo]["main"].get(path)

    def paths(self, repo: str) -> list[str]:
        return sorted(self.fake.repos[repo]["main"])

    def _rows(self, data: bytes | None, model: type) -> list:
        if data is None:
            return []
        scratch = self.fake.root / "read"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "rows.jsonl").write_bytes(data)
        return read_rows(scratch / "rows.jsonl", model)

    def labels(self, language: str = "en") -> list[LabelRow]:
        return self._rows(self.stored(labels_path(self.slug, language, self.mode), EVAL), LabelRow)

    def raws(self, language: str = "en") -> list[RawRow]:
        return self._rows(self.stored(judge_raw_path(self.slug, language, self.mode), RESPONSES), RawRow)
