"""Scripted judge and a results repo holding one checkpoint's generation rows; placeholder text only."""

import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from transformers import BatchEncoding

from driftmap.core import ids
from driftmap.core.ids import judge_raw_path, labels_path, responses_path
from driftmap.core.records import Decoding, GenerationRow, LabelRow, PartialMarker, Ref, RunRecord
from driftmap.core.schemas import SCHEMA_GENERATION
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from driftmap.judge import polyguard
from driftmap.judge.incremental import RawRow, read_rows
from driftmap.scripts import judge as judge_script
from tests.fakes import FakeClock, FakeHub
from tests.fakes.labels import run_record
from tests.fakes.settings import BASE, settings_toml

SHAPES = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "fixtures" / "judge" / "raw_shapes.jsonl")
    .read_text(encoding="utf-8")
    .split("\n")
    if line
]
CANONICAL, SAFE = SHAPES[0]["raw"], SHAPES[1]["raw"]
UNPARSEABLE = [shape["raw"] for shape in SHAPES if shape["labels"] is None]


class FakeJudgeModel:
    """Tokenizer and model in one: answers each chat with the next scripted raw output, then ``default``."""

    device = "cpu"
    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self, raws: Iterable[str] = (), default: str = SAFE) -> None:
        self.padding_side = "right"
        self.raws = list(raws)
        self.default = default
        self.chats: list[list[dict[str, str]]] = []
        self.template_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.before_generate: Callable[[int], None] | None = None
        self._answers: list[str] = []

    @property
    def contents(self) -> list[str]:
        """The user turn of every pair judged so far."""
        return [chat[1]["content"] for chat in self.chats]

    def apply_chat_template(self, chats: list[list[dict[str, str]]], **kwargs: Any) -> BatchEncoding:
        self.template_calls.append({**kwargs, "padding_side": self.padding_side})
        self.chats += chats
        rows = len(chats)
        return BatchEncoding(
            {"input_ids": torch.ones((rows, 3), dtype=torch.long), "attention_mask": torch.ones((rows, 3))}
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append({**kwargs, "batch": len(input_ids)})
        if self.before_generate is not None:
            self.before_generate(len(self.generate_calls))
        first = len(self._answers)
        self._answers += [self.raws.pop(0) if self.raws else self.default for _ in input_ids]
        answer_ids = torch.arange(first + 1, len(self._answers) + 1).unsqueeze(1)
        return torch.cat([input_ids, answer_ids], dim=1)

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> list[str]:
        assert skip_special_tokens
        return [self._answers[int(row[0]) - 1] for row in token_ids]


OWNER = BASE["owner"]
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
RESPONSES = f"{OWNER}/{BASE['repos']['responses']}"
EVAL = f"{OWNER}/{BASE['repos']['eval']}"
JUDGE, SMOL = BASE["judge"]["repo"], BASE["judge"]["smol_repo"]
MODEL = BASE["sources"]["model"]
CHECKPOINT = f"{MODEL}@{'1' * 40}"
SLUG = f"tiny-aya-global--{'1' * 12}"
GENERATION = "generate-full-20261113T080000Z"
RUN_ID = "judge-full-20261113T100000Z"
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


class JudgeWorld:
    """A Hub with the judge repos, one finished generation run's rows, the evaluation dataset and the results repo."""

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
        # the job's working directories go with the test's own
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        self.mode, self.checkpoint, self.slug = mode, ids.parse(checkpoint), ids.slug(checkpoint)
        self.fake = FakeHub(tmp_path / "hub")
        for repo in (JUDGE, SMOL):
            self.fake.add_repo(repo, files={"tokenizer.json": b"{}", "config.json": b"{}", "model.safetensors": b"w"})
        self.rows = {
            (set_, language): generation_rows(set_, language, n, checkpoint=checkpoint)
            for set_, codes in (("harmful", languages), ("xstest", ("en",)), ("benign", languages if benign else ()))
            for language in codes
        }
        files = {responses_path(self.slug, set_, code, mode): jsonl(rows) for (set_, code), rows in self.rows.items()}
        for repo in (RESULTS, RESPONSES, EVAL):
            self.fake.add_repo(repo, repo_type="dataset", private=True, files=files if repo == RESPONSES else {})
            self.fake.authorize(repo)
        self.put_record()
        pins = {"revision": self.fake.heads[JUDGE], "smol_revision": self.fake.heads[SMOL]}
        judge = {**pins, **overrides.pop("judge", {})}
        self.config = settings_toml(tmp_path, languages=list(languages), judge=judge, **overrides)
        self.model = FakeJudgeModel()
        self.loads: list[str] = []
        monkeypatch.setattr(polyguard, "load_judge_tokenizer", lambda directory: self.model)
        monkeypatch.setattr(polyguard, "load_model", self._load)

    def _load(self, directory: str, dtype: str, device: str) -> FakeJudgeModel:
        self.loads.append(f"{dtype} on {device}")
        return self.model

    def put(self, path: str, data: bytes, repo: str = RESULTS) -> None:
        self.fake.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")

    def drop(self, path: str, repo: str) -> None:
        """Remove a file from the repo's head, as a commit would."""
        for revision in ("main", self.fake.heads[repo]):
            del self.fake.repos[repo][revision][path]

    def put_rows(self, set_: str, language: str, rows: Iterable[GenerationRow]) -> None:
        """Replace one responses file, as a later upload of the generation run."""
        self.put(responses_path(self.slug, set_, language, self.mode), jsonl(rows), RESPONSES)
        self.put_record()

    def put_record(self, run_id: str = GENERATION, **changes) -> None:
        """The generation run's record, naming the current head of the responses repo as the home of its rows."""
        fields = {
            "run_id": run_id,
            "mode": self.mode,
            "inputs": [Ref(kind="model", repo=self.checkpoint.repo, revision=self.checkpoint.revision)],
            "outputs": [Ref(kind="dataset", repo=RESPONSES, revision=self.fake.heads[RESPONSES])],
        }
        record = run_record(changes.pop("kind", "generate"), **{**fields, **changes})
        self.put(record_path(run_id), record.model_dump_json(by_alias=True).encode())

    def run(self, *extra: str, run_id: str = RUN_ID, device: str = "cpu", clock: FakeClock | None = None) -> int:
        argv = ["--run-id", run_id, "--timeout-s", "2700", "--code-sha", "a" * 40]
        argv += ["--submitted-at", "2026-11-02T10:00:00Z", "--config", str(self.config)]
        argv += ["--checkpoint", ids.unparse(self.checkpoint), "--generation", GENERATION, *extra]
        if self.mode == "smoke":
            argv += ["--force-flush-after-s", "300"]
        return judge_script.main(argv, client=self.fake, clock=clock or FakeClock(), device=device)

    def stored(self, path: str, repo: str = RESULTS) -> bytes | None:
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

    def record(self, run_id: str = RUN_ID) -> RunRecord:
        return RunRecord.model_validate_json(self.stored(record_path(run_id)))

    def marker(self, run_id: str = RUN_ID) -> PartialMarker:
        return PartialMarker.model_validate_json(self.stored(marker_path(run_id)))

    def downloads(self, repo: str) -> list[dict[str, Any]]:
        calls = [kwargs for method, kwargs in self.fake.calls if method == "snapshot_download"]
        return [kwargs for kwargs in calls if kwargs["repo_id"] == repo]
