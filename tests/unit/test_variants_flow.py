"""The script variants end to end on a fake Hub: built and uploaded, generated, then judged."""

import json
import tempfile

from auditkit.model import CallableModel

from driftmap.core import ids, render
from driftmap.core.hub import Hub
from driftmap.core.records import GenerationRow, LabelRow, RunRecord
from driftmap.core.settings import load_settings
from driftmap.data import variants
from driftmap.jobkit.runrecord import record_path
from driftmap.judge import closed
from driftmap.judge.incremental import read_rows
from driftmap.scripts import generate
from tests.fakes import FakeClock, FakeHub
from tests.fakes.generator import FakeGenerator
from tests.fakes.settings import BASE, settings_toml
from tests.fakes.tokenizers import build

MODEL = BASE["sources"]["model"]
OWNER = BASE["owner"]
EVAL, RESULTS, RESPONSES = (f"{OWNER}/{BASE['repos'][name]}" for name in ("eval", "results", "responses"))
RUN_ID = "generate-full-20261114T100000Z"
ARGV = ["--timeout-s", "5400", "--code-sha", "a" * 40, "--submitted-at", "2026-11-14T10:00:00Z"]
# harmless stand-ins for the prompts of each language
TEXTS = {
    "hi": ["नमस्ते दुनिया", "दाल कैसे बनाते हैं?"],
    "ru": ["Привет, мир!", "Расскажи, как это сделать"],
    "ar": ["مرحبا بالعالم", "كيف حالك اليوم؟"],
}
REWRITTEN = ["mar7aba bil3alam", "kif 7alak el yom?"]
REFUSED = '{"request_harm": "yes", "refusal": "yes", "response_harm": "no"}'


class Refusing:
    """A judge that reads every answer as a refusal."""

    prompt_tokens = completion_tokens = 0

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return REFUSED


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def prompt_files():
    files = {}
    for set_, folder in variants.SETS.items():
        for language, texts in TEXTS.items():
            rows = [
                {"prompt_id": f"{set_}-{language}-{index:03d}", "language": language, "set": set_, "prompt": text}
                | ({"target": f"placeholder answer {index}"} if set_ == "benign" else {"global_or_local": "global"})
                for index, text in enumerate(texts)
            ]
            files[f"prompts/{folder}/{language}.jsonl"] = jsonl(rows)
    return files


def stored(fake, repo, path, model):
    scratch = fake.root / "read"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "rows.jsonl").write_bytes(fake.repos[repo]["main"][path])
    return read_rows(scratch / "rows.jsonl", model)


def test_variants_the_data_module_uploads_are_generated_and_judged_in_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    tokenizer = tmp_path / "tokenizer"
    build().save_pretrained(tokenizer)
    weights = {path.name: path.read_bytes() for path in tokenizer.iterdir()} | {"model.safetensors": b"weights"}
    prefix = render.render(build(), "hello").rendered_prefix_sha256
    manifest = {"schema": "driftmap.manifest/1", "seed": BASE["seed"], "rendered_prefix_sha256": prefix}
    published = prompt_files() | {"manifest.json": (json.dumps(manifest) + "\n").encode()}
    fake = FakeHub(tmp_path / "hub")
    fake.add_repo(MODEL, gated=True, files=weights)
    fake.add_repo(EVAL, repo_type="dataset", private=True, files=published)
    fake.add_repo(RESULTS, repo_type="dataset", private=True)
    for repo in (MODEL, EVAL, RESULTS):
        fake.authorize(repo)
    pinned = fake.heads[EVAL]
    config = settings_toml(tmp_path, languages=list(TEXTS), revisions={MODEL: fake.heads[MODEL], EVAL: pinned})
    settings = load_settings(config)
    # a local build folder with the pinned prompt files and one file of accepted rewrites
    data = tmp_path / "data"
    for path, content in prompt_files().items():
        (data / path).parent.mkdir(parents=True, exist_ok=True)
        (data / path).write_bytes(content)
    rewrites = [
        {"parent_prompt_id": f"harmful-ar-{index:03d}", "prompt": text} for index, text in enumerate(REWRITTEN)
    ]
    (data / "variants_source" / "harmful").mkdir(parents=True)
    (data / "variants_source" / "harmful" / "ar-Latn.jsonl").write_bytes(jsonl(rewrites))

    uploaded = variants.upload(settings, Hub(fake, sleep=lambda seconds: None), data)

    built = [(made.set, made.code, len(made.rows)) for made in uploaded.files if made.rows]
    assert ("harmful", "ar-Latn", 2) in built and ("benign", "hi-Latn", 2) in built and len(built) == 5
    assert uploaded.commit == fake.heads[EVAL] != pinned
    assert sorted(path for path in fake.repos[EVAL]["main"] if path.startswith("variants/")) == sorted(
        ids.variant_prompts(set_, code) for set_, code, _ in built
    )

    monkeypatch.setattr(generate, "load_hf_model", lambda *loaded: CallableModel(FakeGenerator()))
    named = ["--kind", "baseline", "--sets", "variants", "--variants-rev", uploaded.commit]
    argv = ["--run-id", RUN_ID, *ARGV, "--config", str(config), *named]
    assert generate.main(argv, client=fake, clock=FakeClock(), device="cpu") == 0

    checkpoint = f"{MODEL}@{fake.heads[MODEL]}"
    slug = ids.slug(checkpoint)
    answers = {(set_, code): ids.responses_path(slug, set_, code, "full", variants=True) for set_, code, _ in built}
    assert sorted(fake.repos[RESPONSES]["main"]) == sorted(answers.values())
    for (set_, code), path in answers.items():
        asked = [json.loads(line) for line in fake.repos[EVAL]["main"][ids.variant_prompts(set_, code)].splitlines()]
        rows = stored(fake, RESPONSES, path, GenerationRow)
        assert {(row.set, row.language) for row in rows} == {(set_, code)}
        assert [row.prompt_id for row in rows] == [row["prompt_id"] for row in asked]
        assert all(row.status == "ok" and row.rendered_prefix_sha256 == prefix for row in rows)
    record = RunRecord.model_validate_json(fake.repos[RESULTS]["main"][record_path(RUN_ID)])
    assert record.status == "ok" and record.prompts == sum(count for _, _, count in built)
    assert [ref.revision for ref in record.inputs if ref.kind == "dataset"] == [pinned, uploaded.commit]
    # the evaluation dataset got no label of a benign variant
    assert not [path for path in fake.repos[EVAL]["main"] if "labels" in path]

    judge = Refusing()
    closed.judge_checkpoint(
        settings, Hub(fake), judge, checkpoint, mode="full", sets=["variants"], echo=lambda line: None
    )

    harmful = [code for set_, code, _ in built if set_ == "harmful"]
    labelled = sorted(path for path in fake.repos[EVAL]["main"] if "labels" in path)
    assert labelled == sorted(ids.labels_path(slug, code, "full", variants=True) for code in harmful)
    for code in harmful:
        labels = stored(fake, EVAL, ids.labels_path(slug, code, "full", variants=True), LabelRow)
        assert {(row.set, row.language, row.refusal, row.response_harm) for row in labels} == {
            ("harmful", code, "yes", "no")
        }
        assert ids.judge_raw_path(slug, code, "full", variants=True) in fake.repos[RESPONSES]["main"]
    assert judge.calls == sum(count for set_, _, count in built if set_ == "harmful")
