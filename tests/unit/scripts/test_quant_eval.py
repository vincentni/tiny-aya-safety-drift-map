import hashlib
import json
import subprocess
import tempfile
import threading
import urllib.error
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

from driftmap.core import ids, render
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import GenerationRow, LabelRow, PartialMarker, Ref, RunRecord
from driftmap.core.render import token_count
from driftmap.core.settings import load_settings
from driftmap.jobkit.deadline import DeadlineReached
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from driftmap.judge import units
from driftmap.scripts import generate, quant_eval
from driftmap.scripts.quant_eval import FALLBACK, Tools, cuda_library_path
from driftmap.table.build import NOT_COMPARABLE, column_of
from tests.fakes import FakeClock, FakeHub, FakeTransport
from tests.fakes.labels import run_record
from tests.fakes.release import BINARY_ROOT, RELEASE, FakeOpener, archives, pages
from tests.fakes.release import pinned as pinned_release
from tests.fakes.settings import BASE, settings_toml
from tests.fakes.tokenizers import build

OWNER = BASE["owner"]
MODEL = BASE["sources"]["model"]
REPAIRED = f"{OWNER}/{BASE['repos']['repaired']}"
GGUF = f"{OWNER}/{BASE['repos']['gguf']}"
EVAL = f"{OWNER}/{BASE['repos']['eval']}"
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
RESPONSES = f"{OWNER}/{BASE['repos']['responses']}"
LANGUAGES = BASE["languages"]
FOLDERS = {"harmful": "harmful", "xstest": "xstest", "benign": "benign_heldout"}
# the order the job works in
UNITS = [("harmful", code) for code in LANGUAGES] + [("xstest", "en")] + [("benign", code) for code in LANGUAGES]
ROWS = 4
FILE = "model-q4_k_m.gguf"
Q4 = b"GGUF q4_k_m weights"
Q4_SHA = hashlib.sha256(Q4).hexdigest()
RUN_ID = "quant_eval-full-20261102T100000Z"
LATER = "quant_eval-full-20261104T100000Z"
ARGV = ["--timeout-s", "7200", "--code-sha", "a" * 40, "--submitted-at", "2026-11-02T10:00:00Z"]
SMOKE_ARGS = ["--force-flush-after-s", "300"]
SMOKE = "smoke/"
EMPTY = {"content": "  ", "stop_type": "eos", "truncated": False}


def prompt_rows(set_, language, count=ROWS):
    rows = []
    for index in range(count):
        row = {"prompt_id": f"{set_}-{language}-{index:03d}", "prompt": f"hello {set_} {language} {index}"}
        if set_ == "benign":
            row["target"] = f"bonjour canary {index}"
        rows.append(row)
    return rows


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def rendered(prompt):
    return render.render(build(), prompt).rendered_prompt


def responses_at(slug, set_, code, tree=""):
    return f"{tree}responses/{slug}/{FOLDERS[set_]}/{code}.jsonl"


def labels_at(slug, code, tree=""):
    return f"{tree}labels/{slug}/benign/{code}.jsonl"


class Server(FakeTransport):
    """Answers a prompt with a short reply and the tokenizer's count of it; scripted replies come first."""

    def __init__(self):
        super().__init__()
        self.tokenizer = build()
        self.extra_tokens = 0
        self.props = {"total_slots": 4, "default_generation_settings": {"n_ctx": 2048}}
        # completions answered before the server goes away; the hook marks it
        self.dies_after = None
        self.on_death = None
        self._served = 0
        self._counting = threading.Lock()

    def _completion(self, path, body):
        prompt = body["prompt"]
        with self._counting:
            self._served += 1
            dead = self.dies_after is not None and self._served > self.dies_after
            if dead and self.health == "ok":
                self.health = "refused"
                self.on_death()
        if dead:
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        if self.replies_for.get(prompt):
            return super()._completion(path, body)
        with self._counting:
            evaluated = token_count(self.tokenizer, prompt) + self.extra_tokens
        return {
            "content": "bonjour canary reply",
            "tokens_evaluated": evaluated,
            "stop_type": "eos",
            "truncated": False,
        }

    def prompts(self, start=0):
        return [body["prompt"] for _, path, body in self.calls[start:] if path == "/completion"]

    def count(self, path):
        return sum(called == path for _, called, _ in self.calls)


class Process:
    def __init__(self, argv, said, stubborn, interrupt, **kwargs):
        self.argv, self.kwargs, self.stubborn = list(argv), kwargs, stubborn
        # raised once, inside the first wait
        self.interrupt = interrupt
        self.returncode = None
        self.signals = []
        self.say(said)

    def say(self, text):
        self.kwargs["stdout"].write(text)
        self.kwargs["stdout"].flush()

    def die(self, code, said):
        self.returncode = code
        self.say(said)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")

    def wait(self, timeout=None):
        self.signals.append(("wait", timeout))
        if self.interrupt is not None:
            error, self.interrupt = self.interrupt, None
            raise error
        killed = "kill" in self.signals
        if self.stubborn and not killed:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.returncode is None:
            self.returncode = -9 if killed else -15

    def kill(self):
        self.signals.append("kill")


class Popen:
    """Stand-in for ``subprocess.Popen``; no program is started."""

    def __init__(self):
        self.started = []
        self.said = b"main: server is listening on http://127.0.0.1:8080\n"
        self.stubborn = False
        self.interrupt = None
        self.error = None

    def __call__(self, argv, **kwargs):
        if self.error:
            raise self.error
        self.started.append(Process(argv, self.said, self.stubborn, self.interrupt, **kwargs))
        return self.started[-1]


class CorruptingHub(FakeHub):
    """The GGUF arrives as other bytes than the Hub lists."""

    def hf_hub_download(self, repo_id, filename, **kwargs):
        path = super().hf_hub_download(repo_id, filename, **kwargs)
        if filename == FILE:
            Path(path).write_bytes(Q4[:-1] + b"S")
        return path


class TickingClock(FakeClock):
    """Half a second passes at every reading."""

    def monotonic(self):
        self.advance(0.5)
        return super().monotonic()


@dataclass
class World:
    fake: FakeHub
    config: Path
    gguf: str
    repaired: str
    server: Server
    popen: Popen
    opener: FakeOpener
    gpu_used: object = None

    def run(self, *extra, run_id=RUN_ID, device="cpu", clock=None):
        argv = ["--run-id", run_id, *ARGV, "--config", str(self.config), *extra]
        tools = Tools(self.opener, self.popen, self.server, FakeClock(), self.gpu_used)
        return quant_eval.main(argv, client=self.fake, clock=clock or FakeClock(), device=device, tools=tools)

    def head(self, repo):
        return self.fake.heads[repo]

    @property
    def checkpoint(self):
        return f"{self.gguf}@{self.head(self.gguf)}#{FILE}"

    @property
    def slug(self):
        return f"{self.gguf.partition('/')[2]}--{self.head(self.gguf)[:12]}"

    def stored(self, repo, path):
        return self.fake.repos[repo]["main"].get(path)

    def put(self, repo, path, data):
        self.fake.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")

    def add_record(self, record):
        self.put(RESULTS, record_path(record.run_id), record.model_dump_json(by_alias=True).encode())

    def record(self, run_id=RUN_ID):
        return RunRecord.model_validate_json(self.stored(RESULTS, record_path(run_id)))

    def notes(self, run_id=RUN_ID):
        return " | ".join(self.record(run_id).notes)

    def marker(self, run_id=RUN_ID):
        return PartialMarker.model_validate_json(self.stored(RESULTS, marker_path(run_id)))

    def rows(self, set_, language, tree=""):
        lines = self.stored(RESPONSES, responses_at(self.slug, set_, language, tree)).decode().split("\n")
        return [GenerationRow.model_validate_json(line) for line in lines if line]

    def uploads(self, repo):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "upload_file" and kwargs["repo_id"] == repo]
        return [kwargs["path_in_repo"] for kwargs in calls]

    def gguf_downloads(self):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "hf_hub_download"]
        return [kwargs for kwargs in calls if kwargs["filename"] == FILE]

    def kill_server_after(self, completions, code, said):
        self.server.dies_after = completions
        self.server.on_death = lambda: self.popen.started[0].die(code, said)

    def script(self, prompt, reply):
        """One scripted reply; it counts the prompt as the tokenizer does unless it says otherwise."""
        text = rendered(prompt)
        self.server.replies_for[text] = deque([{"tokens_evaluated": token_count(build(), text), **reply}])


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    # the job's scratch directories go with the test's
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def make(mode="full", hub=FakeHub, files=None, described=None, quant_record=True, settings=None):
        suffix = "-smoke" if mode == "smoke" else ""
        repaired, gguf = REPAIRED + suffix, GGUF + suffix
        source = tmp_path / "source"
        build().save_pretrained(source)
        tokenizer_files = {path.name: path.read_bytes() for path in source.iterdir()}
        fake = hub(tmp_path / "hub")
        fake.add_repo(repaired, private=True, files=tokenizer_files | {"model.safetensors": b"repaired weights"})
        record = {
            "schema": "driftmap.gguf/1",
            "llamacpp_tag": RELEASE["tag"],
            "tarball": RELEASE["source_tarball"],
            "tarball_sha256": RELEASE["sha256"]["source_tarball"],
            "quant": "q4_k_m",
            "sha256": Q4_SHA,
            "bytes": len(Q4),
            "source_repo": repaired,
            "source_revision": fake.heads[repaired],
        }
        gguf_files = {FILE: Q4, "gguf.json": json.dumps(record | (described or {})).encode()}
        fake.add_repo(gguf, private=True, files={name: data for name, data in gguf_files.items() if data is not None})
        prefix = render.render(build(), "hello").rendered_prefix_sha256
        manifest = {"schema": "driftmap.manifest/1", "seed": BASE["seed"], "rendered_prefix_sha256": prefix}
        prompts = {f"prompts/{FOLDERS[set_]}/{code}.jsonl": jsonl(prompt_rows(set_, code)) for set_, code in UNITS}
        prompts["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        fake.add_repo(EVAL, repo_type="dataset", private=True, files=prompts | (files or {}))
        fake.add_repo(RESULTS, repo_type="dataset", private=True)
        for repo in (repaired, gguf, EVAL, RESULTS):
            fake.authorize(repo)
        release_files = archives()
        config = settings_toml(tmp_path, **pinned_release(release_files), **(settings or {}))
        world = World(fake, config, gguf, repaired, Server(), Popen(), FakeOpener(pages(release_files)))
        if quant_record:
            pushed = Ref(kind="model", repo=gguf, revision=fake.heads[gguf], sha256=Q4_SHA)
            world.add_record(run_record("quant", run_id=f"quant-{mode}-1", mode=mode, outputs=[pushed]))
        return world

    return make


@pytest.fixture
def world(make_world):
    return make_world()


def test_every_unit_is_answered_by_the_served_gguf_and_recorded_for_the_judge(world, tmp_path):
    read_at, source_at = world.head(EVAL), world.head(REPAIRED)
    assert world.run(clock=TickingClock()) == 0

    assert world.uploads(RESPONSES) == [responses_at(world.slug, set_, code) for set_, code in UNITS]
    for set_, code in UNITS:
        rows = world.rows(set_, code)
        asked = prompt_rows(set_, code)
        assert [(row.prompt_id, row.prompt) for row in rows] == [(row["prompt_id"], row["prompt"]) for row in asked]
        assert {(row.checkpoint, row.set, row.language, row.status) for row in rows} == {
            (world.checkpoint, set_, code, "ok")
        }
        assert {(row.decoding.runtime, row.response, row.stop_reason) for row in rows} == {
            ("llama-server", "bonjour canary reply", "eos")
        }
    assert world.uploads(EVAL) == [labels_at(world.slug, code) for code in LANGUAGES]
    lines = world.stored(EVAL, labels_at(world.slug, "hi")).decode().split("\n")
    labels = [LabelRow.model_validate_json(line) for line in lines if line]
    assert all(0 < label.chrf <= 1 and label.checkpoint == world.checkpoint for label in labels) and len(labels) == ROWS

    # rendered text as sent: a probe per language, then every prompt
    sent = world.server.prompts()
    assert len(sent) == len(LANGUAGES) + ROWS * len(UNITS)
    assert sent[: len(LANGUAGES)] == [rendered(f"hello harmful {code} 0") for code in LANGUAGES]
    assert set(sent) == {rendered(row["prompt"]) for set_, code in UNITS for row in prompt_rows(set_, code)}

    record = world.record()
    pinned = load_settings(world.config).llamacpp_release
    assert (record.status, record.kind, record.mode, record.prompts) == ("ok", "quant_eval", "full", ROWS * len(UNITS))
    assert (record.bos_check, record.upload_mb_s) == (True, None) and record.tokens_per_s > 0
    assert record.inputs == [
        Ref(kind="model", repo=GGUF, revision=world.head(GGUF), sha256=Q4_SHA),
        Ref(kind="dataset", repo=EVAL, revision=read_at),
        Ref(kind="model", repo=REPAIRED, revision=source_at),
        Ref(kind="binary", repo=RELEASE["cuda_asset"], revision="b11960", sha256=pinned.sha256.cuda_asset),
    ]
    assert {ref.repo: ref.revision for ref in record.outputs} == {repo: world.head(repo) for repo in (RESPONSES, EVAL)}
    assert "llama-server: 4 slots, 2048 tokens of context each" in record.notes
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert json.loads(conditions)["decoding"] == {
        "greedy": True,
        "max_new_tokens": 256,
        "runtime": "llama-server",
        "cache_prompt": False,
    }
    assert [url for url, _ in world.opener.calls] == [RELEASE["cuda_asset"]]
    assert world.popen.started[0].signals == ["terminate", ("wait", 1.0)]

    # the judge reads these rows by --checkpoint <repo>@<sha>#<file>; the table maps it to the q4 column
    hub = Hub(world.fake)
    settings = load_settings(world.config)
    generated = RepoRef(RESPONSES, world.head(RESPONSES))
    rows_dir = Path(hub.snapshot_download(generated, tmp_path / "judge-rows", repo_type="dataset"))
    inputs = units.read_inputs(rows_dir, ids.slug(world.checkpoint), units.units(settings), "full", None)
    assert [len(rows) for rows in inputs.values()] == [ROWS] * 7
    assert column_of(world.checkpoint, settings, "full") == "repaired_q4"


def test_a_gguf_that_differs_from_its_record_never_reaches_the_server(make_world):
    """A download that hashes unlike gguf.json exits 12 before the server starts: zero /health calls (TS-QNT-4)."""
    world = make_world(hub=CorruptingHub)
    assert world.run() == 12
    record = world.record()
    assert record.status == "failed" and "gguf.json says" in world.notes()
    assert world.server.count("/health") == 0 and world.server.calls == [] and world.popen.started == []
    assert world.uploads(RESPONSES) == []

    # a record unlike the Hub's own hash stops before any download
    stale = make_world(described={"sha256": hashlib.sha256(b"an earlier file").hexdigest()})
    assert stale.run() == 12
    assert "the Hub holds" in stale.notes()
    assert stale.gguf_downloads() == [] and stale.opener.calls == [] and stale.popen.started == []


def test_the_server_runs_on_loopback_and_a_unit_over_five_percent_unscored_fails_the_column(make_world):
    """llama-server gets exact arguments, no token, no exposed port; 6 % unscored in a unit exits 13 (TS-QNT-5)."""
    world = make_world(files={"prompts/harmful/fr.jsonl": jsonl(prompt_rows("harmful", "fr", 50))})
    for index in (7, 8, 9):
        world.script(f"hello harmful fr {index}", EMPTY)

    assert world.run() == 13

    [process] = world.popen.started
    assert process.argv[0].endswith(f"cuda_asset/{BINARY_ROOT}/llama-server")
    gguf = process.argv[2]
    assert Path(gguf).name == FILE and Path(gguf).read_bytes() == Q4
    assert process.argv[1:] == [
        *["-m", gguf, "-ngl", "all", "--parallel", "4", "-c", "8192"],
        *["--host", "127.0.0.1", "--port", "8080", "--no-slots"],
    ]
    assert not [part for part in process.argv if "expose" in part]
    env = process.kwargs["env"]
    assert "HF_TOKEN" not in env and not [value for value in env.values() if "hf_dummy" in value]
    assert process.kwargs["stdin"] == subprocess.DEVNULL and process.signals[0] == "terminate"

    record = world.record()
    assert record.status == "failed"
    assert "harmful/fr: 3 of 50 rows unscored (empty 3), over the 5 % limit" in world.notes()
    # the server is alive and well: the rows are the cause
    assert FALLBACK not in world.notes()
    # the failed unit leaves no file and is not marked done
    assert world.uploads(RESPONSES) == [responses_at(world.slug, "harmful", "en")]
    assert world.marker().done == [responses_at(world.slug, "harmful", "en")]


def test_exactly_five_percent_unscored_passes(make_world):
    world = make_world(files={"prompts/harmful/fr.jsonl": jsonl(prompt_rows("harmful", "fr", 20))})
    world.script("hello harmful fr 7", EMPTY)
    assert world.run() == 0
    statuses = [row.status for row in world.rows("harmful", "fr")]
    assert statuses.count("unscored") == 1 and len(statuses) == 20 and world.record().bos_check is True


def test_a_second_bos_stops_the_job_before_any_unit(world):
    world.server.extra_tokens = 1
    assert world.run() == 12
    record = world.record()
    assert (record.status, record.bos_check, record.prompts) == ("failed", False, 0)
    assert "another token count than the tokenizer" in world.notes()
    # the diagnostic asks the server for its ids
    assert world.server.count("/tokenize") == len(LANGUAGES) and world.uploads(RESPONSES) == []
    assert world.popen.started[0].signals[0] == "terminate"


def test_a_prompt_counted_differently_inside_a_unit_is_noted_and_fails_the_bos_check(world):
    off = {"content": "bonjour", "tokens_evaluated": 3, "stop_type": "eos", "truncated": False}
    world.script("hello harmful es 2", off)
    assert world.run() == 0
    record = world.record()
    assert record.bos_check is False
    assert "harmful/es: the server evaluated another token count than the tokenizer for 1 prompts" in record.notes


def test_a_server_that_never_answers_fails_the_column_and_is_stopped(make_world):
    world = make_world()
    world.server.health = "refused"
    world.popen.said = b"error while loading shared libraries: libcudart.so.12: cannot open shared object file\n"
    assert world.run() == 13
    notes = world.notes()
    assert "not healthy after 180 s" in notes and "libcudart.so.12" in notes and FALLBACK in notes
    assert world.popen.started[0].signals == ["terminate", ("wait", 1.0)] and world.server.prompts() == []

    full = make_world()
    full.server.health = "refused"
    full.popen.said = b"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2000.00 MiB: out of memory\n"
    # the line that names the cause is rarely the last one
    full.popen.said += b"llama_model_load: error loading model: unable to allocate CUDA0 buffer\n" * 8
    assert full.run() == 11
    assert full.record().status == "oom" and "out of memory" in full.record().oom

    absent = make_world()
    absent.popen.error = OSError(8, "Exec format error")
    assert absent.run() == 13
    assert "llama-server did not start" in absent.notes() and absent.server.calls == []


def test_a_server_that_ignores_the_stop_is_killed(world):
    world.popen.stubborn = True
    assert world.run() == 0
    assert world.popen.started[0].signals == ["terminate", ("wait", 1.0), "kill", ("wait", 0.5)]


def test_a_stop_signal_during_the_wait_still_kills_the_server(world):
    world.popen.interrupt = DeadlineReached("stopped by SIGTERM", urgent=True)
    assert world.run() == 10
    assert world.popen.started[0].signals == ["terminate", ("wait", 1.0), "kill", ("wait", 0.5)]
    assert world.record().status == "partial"


@pytest.mark.parametrize("answered", [0, 2])
def test_a_server_that_dies_inside_a_unit_is_reported_with_its_status_and_log(world, answered):
    said = b"terminate called after throwing an instance of 'std::runtime_error'\n"
    world.kill_server_after(len(LANGUAGES) + answered, 139, said)
    assert world.run() == 13
    notes = world.notes()
    assert "llama-server exited 139" in notes and "std::runtime_error" in notes and FALLBACK in notes
    assert world.uploads(RESPONSES) == [] and world.marker().done == []


@pytest.mark.parametrize("answered", [0, 2])
def test_a_server_out_of_memory_inside_a_unit_is_an_out_of_memory_stop(world, answered):
    said = b"CUDA error: out of memory\n" + b"srv    operator(): cleaning up before exit...\n" * 12
    world.kill_server_after(len(LANGUAGES) + ROWS + answered, 134, said)
    assert world.run() == 11
    record = world.record()
    assert record.status == "oom" and "CUDA error: out of memory" in record.oom
    assert world.marker().done == [responses_at(world.slug, "harmful", "en")]


def test_a_resumed_run_keeps_a_failed_token_count_check(make_world):
    world = make_world()
    off = {"content": "bonjour", "tokens_evaluated": 3, "stop_type": "eos", "truncated": False}
    world.script("hello harmful en 2", off)
    world.script("hello harmful es 1", EMPTY)
    assert world.run() == 13
    assert world.record().bos_check is False

    assert world.run("--resume", RUN_ID, run_id=LATER) == 0
    record = world.record(LATER)
    # harmful/en is not generated again, so its failed count still stands
    assert record.bos_check is False
    assert "harmful/en: the server evaluated another token count than the tokenizer for 1 prompts" in record.notes

    # a run stopped at the probes kept no unit to carry
    fresh = make_world()
    fresh.server.extra_tokens = 1
    assert fresh.run() == 12
    fresh.server.extra_tokens = 0
    assert fresh.run("--resume", RUN_ID, run_id=LATER) == 0
    assert fresh.record(LATER).bos_check is True


def test_a_model_left_off_the_gpu_fails_the_column(make_world):
    idle = make_world()
    idle.gpu_used = lambda: 300
    assert idle.run() == 13
    assert "did not load the model on the GPU" in idle.notes() and idle.server.prompts() == []

    readings = iter([300, 300 + len(Q4)])
    loaded = make_world()
    loaded.gpu_used = lambda: next(readings)
    assert loaded.run() == 0
    assert "llama-server holds 0.00 GiB of GPU memory" in loaded.record().notes


def test_a_resumed_run_generates_only_the_units_left_on_the_same_file(world):
    world.script("hello harmful es 1", EMPTY)
    assert world.run() == 13
    done = [responses_at(world.slug, "harmful", code) for code in ("en", "fr")]
    assert world.marker().done == done
    first, served = world.head(GGUF), len(world.server.calls)
    slug = world.slug
    # a later quant run must not move a resumed column
    world.fake.upload_file(path_or_fileobj=b"# card", path_in_repo="README.md", repo_id=GGUF)
    newer = Ref(kind="model", repo=GGUF, revision=world.head(GGUF), sha256=Q4_SHA)
    world.add_record(run_record("quant", run_id="quant-full-2", outputs=[newer]))

    assert world.run("--resume", RUN_ID, run_id=LATER) == 0

    record = world.record(LATER)
    assert record.inputs[0].revision == first and record.prompts == ROWS * (len(UNITS) - 2)
    again = world.server.prompts(served)
    assert not [prompt for prompt in again if "harmful en" in prompt or "harmful fr" in prompt]
    assert len(again) == len(LANGUAGES) + ROWS * (len(UNITS) - 2)
    assert responses_at(slug, "benign", "ar") in world.uploads(RESPONSES)

    assert world.run("--resume", "quant_eval-full-unknown", run_id="quant_eval-full-third") == 12


def test_a_smoke_run_takes_the_limit_and_the_smoke_repos(make_world):
    world = make_world(mode="smoke")
    assert world.run("--limit", "2", *SMOKE_ARGS, clock=TickingClock()) == 0
    record = world.record()
    assert (record.status, record.mode, record.prompts, record.bos_check) == ("ok", "smoke", 2 * len(UNITS), True)
    assert record.inputs[0].repo == f"{GGUF}-smoke" and record.inputs[2].repo == f"{REPAIRED}-smoke"
    assert world.uploads(RESPONSES) == [responses_at(world.slug, set_, code, SMOKE) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(world.slug, code, SMOKE) for code in LANGUAGES]
    assert record.upload_mb_s > 0

    assert make_world().run("--limit", "2") == 2


def test_the_column_is_compared_with_the_baseline_run_apart_from_its_runtime(world):
    prefix = render.render(build(), "hello").rendered_prefix_sha256
    conditions = {
        "seed": BASE["seed"],
        "decoding": {"greedy": True, "max_new_tokens": 256, "runtime": "transformers", "cache_prompt": False},
        "rendered_prefix_sha256": prefix,
        "manifest_sha256": hashlib.sha256(world.stored(EVAL, "manifest.json")).hexdigest(),
    }
    baseline = Ref(kind="model", repo=MODEL, revision="1" * 40)

    def add(run_id, **changes):
        note = generate.CONDITIONS + json.dumps(conditions | changes, sort_keys=True)
        world.add_record(run_record("generate", run_id=run_id, inputs=[baseline], notes=[note]))

    add("generate-full-1")
    assert world.run() == 0
    assert "same conditions as baseline run generate-full-1" in world.record().notes

    add("generate-full-2", seed=7)
    assert world.run(run_id=LATER) == 0
    assert NOT_COMPARABLE + "seed differs from baseline" in world.record(LATER).notes


@pytest.mark.parametrize(
    ("change", "argv", "reason"),
    [
        ({}, ["--gguf", f"{REPAIRED}@{'b' * 40}#{FILE}"], f"--gguf must be {GGUF}@<sha>#<file>"),
        ({}, ["--gguf", f"{GGUF}@{'b' * 40}"], f"--gguf must be {GGUF}@<sha>#<file>"),
        ({}, ["--gguf", f"{GGUF}@{'b' * 40}#{FILE}"], "holds no gguf.json"),
        ({"quant_record": False}, [], "no finished quant record"),
        ({"described": {"llamacpp_tag": "b9999"}}, [], "was made with llama.cpp b9999"),
        ({"described": {"source_repo": f"{OWNER}/tiny-aya-global-drift-merged"}}, [], "was converted from"),
        ({"described": {"source_revision": "main"}}, [], "names no pinned source revision"),
        ({"settings": {"decoding": {"max_new_tokens": 128}}}, [], "generates 256 tokens"),
        ({"files": {"manifest.json": b'{"rendered_prefix_sha256": "0"}'}}, [], "is not the evaluation set's"),
    ],
)
def test_bad_inputs_are_refused_before_the_release_or_the_gguf_is_downloaded(make_world, change, argv, reason):
    world = make_world(**change)
    assert world.run(*argv) == 12
    assert reason in world.notes()
    assert world.opener.calls == [] and world.gguf_downloads() == [] and world.popen.started == []


def test_a_slot_too_small_for_a_prompt_and_its_reply_is_refused(world):
    world.server.props = {"total_slots": 4, "default_generation_settings": {"n_ctx": 1024}}
    assert world.run() == 12
    assert "a slot holds 1024 tokens; a prompt and its reply can take 1956" in world.notes()
    assert world.server.prompts() == []

    unknown = World(world.fake, world.config, world.gguf, world.repaired, Server(), Popen(), world.opener)
    unknown.server.props = {"total_slots": 4}
    assert unknown.run(run_id=LATER) == 0
    assert "llama-server did not report its per-slot context" in unknown.record(LATER).notes


def test_a_job_without_a_gpu_stops_at_once(world):
    assert world.run(device="cuda") == 12
    assert "no CUDA device" in world.notes() and world.opener.calls == []


def test_the_cuda_libraries_beside_torch_come_first_on_the_loader_path(tmp_path):
    torch_dir = tmp_path / "site-packages" / "torch"
    for name in ("cublas", "cuda_runtime"):
        (tmp_path / "site-packages" / "nvidia" / name / "lib").mkdir(parents=True)
    packages = tmp_path / "site-packages"
    expected = [packages / "nvidia" / "cublas" / "lib", packages / "nvidia" / "cuda_runtime" / "lib", torch_dir / "lib"]
    assert cuda_library_path(torch_dir, "/usr/local/nvidia/lib").split(":") == [
        *map(str, expected),
        "/usr/local/nvidia/lib",
    ]
    assert cuda_library_path(torch_dir, None).split(":") == list(map(str, expected))
