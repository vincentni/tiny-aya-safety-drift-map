import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from auditkit.model import CallableModel
from safetensors.torch import load as load_tensors
from transformers import Cohere2Config, Cohere2ForCausalLM

from driftmap.core import render
from driftmap.core.records import AlphaScreen, GenerationRow, LabelRow, PartialMarker, Ref, RunRecord
from driftmap.core.schemas import SCHEMA_LABEL
from driftmap.core.settings import load_settings
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from driftmap.repair import screen_gen, select, wiseft
from driftmap.scripts import repair
from tests.fakes import FakeClock, FakeHub
from tests.fakes.labels import judge_config, run_record
from tests.fakes.safetune import FakeSafeTune
from tests.fakes.settings import BASE, settings_toml
from tests.fakes.tokenizers import BOS_ID, SAFETY, build, served_template

OWNER = BASE["owner"]
MODEL = BASE["sources"]["model"]
MERGED = f"{OWNER}/{BASE['repos']['merged']}"
REPAIRED = f"{OWNER}/{BASE['repos']['repaired']}"
EVAL = f"{OWNER}/{BASE['repos']['eval']}"
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
RESPONSES = f"{OWNER}/{BASE['repos']['responses']}"
JUDGE, SMOL = BASE["judge"]["repo"], BASE["judge"]["smol_repo"]
LANGUAGES = BASE["languages"]
FOLDERS = {"harmful": "harmful", "xstest": "xstest", "benign": "benign_heldout"}
UNITS = [("harmful", code) for code in LANGUAGES] + [("xstest", "en")] + [("benign", code) for code in LANGUAGES]
# more than the screen takes from a file
ROWS = 25
SCREEN = 20
PAIRS = SCREEN * len(LANGUAGES)
ALPHAS = [0.1, 0.25, 0.4, 0.5]
RUN_ID = "repair-full-20261113T100000Z"
LATER = "repair-full-20261113T120000Z"
LAST = "repair-full-20261113T140000Z"
ARGV = ["--timeout-s", "7200", "--code-sha", "a" * 40, "--submitted-at", "2026-11-13T10:00:00Z"]
SMOKE_ARGS = ["--force-flush-after-s", "300", "--limit", "5"]
PROJECTION = "model.layers.0.self_attn.q_proj.weight"
OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB"
# the fake backend answers a benign prompt with its target
DRIFTED_CHRF = 1.0
TARGET = "bonjour canary"
ON_CPU = "embeddings interpolated on CPU after an out-of-memory apply"


def prompt_rows(set_, language, count=ROWS):
    rows = [
        {"prompt_id": f"{set_}-{language}-{index:03d}", "prompt": f"hello {set_} {language} {index}"}
        for index in range(count)
    ]
    for row in rows if set_ == "benign" else ():
        row["target"] = TARGET
    return rows


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def dump(rows):
    return "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows).encode()


def slug_of(repo, revision):
    return f"{repo.partition('/')[2]}--{revision[:12]}"


def weights_sha(files):
    return hashlib.sha256(hashlib.sha256(files["model.safetensors"]).hexdigest().encode()).hexdigest()


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Files of two tiny bf16 checkpoints that share a tokenizer: seed 1 is the aligned model, seed 0 the drifted."""
    config = Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=BOS_ID,
        eos_token_id=3,
    )
    files = {}
    for seed in (0, 1):
        directory = tmp_path_factory.mktemp(f"checkpoint-{seed}")
        torch.manual_seed(seed)
        Cohere2ForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)
        build().save_pretrained(directory)
        files[seed] = {path.name: path.read_bytes() for path in directory.iterdir()}
    return files


class CorruptingHub(FakeHub):
    """The merged weights arrive as other bytes than the Hub lists."""

    def snapshot_download(self, repo_id, **kwargs):
        target = super().snapshot_download(repo_id, **kwargs)
        shard = Path(target) / "model.safetensors"
        if repo_id.startswith(MERGED) and shard.is_file() and kwargs.get("allow_patterns") is None:
            shard.write_bytes(shard.read_bytes()[:-1] + b"\x00")
        return target


class NoModelPushHub(FakeHub):
    """Folder uploads are refused; everything else goes through."""

    def upload_folder(self, **kwargs):
        self._raise(403, f"https://huggingface.co/api/models/{kwargs['repo_id']}/commit/main")


class World:
    def __init__(
        self, tmp_path, monkeypatch, checkpoints, *, mode="full", hub=FakeHub, settings=None, fine_tune="standard"
    ):
        (tmp_path / "scratch").mkdir()
        # the job's working directories go with the test's own
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        self.mode = mode
        # the other fine-tune was trained, and is screened here, with no safety preamble
        other = fine_tune != "standard"
        self.suffix = ("-nopre" if other else "") + ("-smoke" if mode == "smoke" else "")
        self.tree = ("smoke/" if mode == "smoke" else "") + ("nopreamble/" if other else "")
        self.screen_file = f"{self.tree}alpha_screen{'-' + fine_tune if other else ''}.json"
        self.merged, self.repaired = MERGED + self.suffix, REPAIRED + self.suffix
        self.aligned_files, self.drifted_files = checkpoints[1], checkpoints[0]
        tokenizer = build()
        if other:
            # both checkpoints carry a tokenizer whose template opens with a safety paragraph, as the served one does
            tokenizer = build(template=served_template())
            tokenizer.save_pretrained(tmp_path / "tokenizer")
            files = {path.name: path.read_bytes() for path in (tmp_path / "tokenizer").iterdir()}
            self.aligned_files, self.drifted_files = (held | files for held in (checkpoints[1], checkpoints[0]))
        self.fake = fake = hub(tmp_path / "hub")
        fake.add_repo(MODEL, gated=True, files=self.aligned_files)
        fake.add_repo(self.merged, private=True, files=self.drifted_files)
        for repo in (JUDGE, SMOL):
            fake.add_repo(repo, files={"tokenizer.json": b"{}", "config.json": b"{}", "model.safetensors": b"w"})
        self.prefix = render.render(tokenizer, "hello").rendered_prefix_sha256
        manifest = {"schema": "driftmap.manifest/1", "seed": BASE["seed"], "rendered_prefix_sha256": self.prefix}
        prompts = {f"prompts/{FOLDERS[set_]}/{code}.jsonl": jsonl(prompt_rows(set_, code)) for set_, code in UNITS}
        prompts["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        fake.add_repo(EVAL, repo_type="dataset", private=True, files=prompts)
        for repo in (RESULTS, RESPONSES):
            fake.add_repo(repo, repo_type="dataset", private=True)
        for repo in (MODEL, self.merged, EVAL, RESULTS, RESPONSES):
            fake.authorize(repo)
        self.drifted = f"{self.merged}@{fake.heads[self.merged]}"
        self.aligned = f"{MODEL}@{fake.heads[MODEL]}"
        # the prompts are pinned; labels land on later commits
        revisions = {MODEL: fake.heads[MODEL], EVAL: fake.heads[EVAL]}
        self.config = settings_toml(tmp_path, revisions=revisions, **(settings or {}))
        self.settings = load_settings(self.config)
        self.add_drift_record()
        self.put_drifted_chrf()

        self.events = []
        self.degraded = set()
        # alpha -> how many benign prompts it answers with nothing
        self.silent = {}
        self.weights = {}
        # called with the alpha at every generation batch
        self.on_generate = lambda alpha: None
        self.safetune = FakeSafeTune().install(monkeypatch)
        self.safetune.calls = self.events
        monkeypatch.setattr(screen_gen, "backend", self._backend)
        real_load = wiseft.load

        def load(directory, device):
            self.events.append(("load", Path(directory).name, device, self.paths(RESPONSES)))
            return real_load(directory, device)

        monkeypatch.setattr(wiseft, "load", load)

    def _backend(self, model, tokenizer, name, batch_size):
        alpha = float(name.split("+")[1].removeprefix("wiseft-"))
        self.weights[alpha] = model.state_dict()[PROJECTION].clone()
        self.events.append(("backend", name, batch_size))

        def reply(prompts):
            self.events.append(("generate", alpha, len(prompts)))
            self.on_generate(alpha)
            benign = "x" if alpha in self.degraded else TARGET
            replies = []
            for prompt in prompts:
                if " benign " not in prompt:
                    replies.append(prompt)
                elif self.silent.get(alpha, 0) > 0:
                    self.silent[alpha] -= 1
                    replies.append("")
                else:
                    replies.append(benign)
            return replies

        return CallableModel(reply)

    def put(self, repo, path, data):
        self.fake.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")

    def add_drift_record(self, sha=None, **changes):
        sha = sha or weights_sha(self.drifted_files)
        pushed = Ref(kind="model", repo=self.merged, revision=self.fake.heads[self.merged], sha256=sha)
        fields = {"run_id": f"drift-{self.mode}", "mode": self.mode, "merged_sha256": sha, "outputs": [pushed]}
        fields |= {"notes": ["preamble: strip"]} if "-nopre" in self.suffix else {}
        record = run_record("drift", **{**fields, **changes})
        self.put(RESULTS, record_path(record.run_id), record.model_dump_json(by_alias=True).encode())

    def _label(self, checkpoint, row, set_, language, **fields):
        blank = {"request_harm": "n/a", "refusal": "n/a", "response_harm": "n/a", "judge": None, "chrf": None}
        return LabelRow(
            schema=SCHEMA_LABEL,
            checkpoint=checkpoint,
            language=language,
            set=set_,
            prompt_id=row["prompt_id"],
            prompt=row["prompt"],
            rendered_prefix_sha256=self.prefix,
            raw_output_present=True,
            **{**blank, **fields},
        )

    def put_drifted_chrf(self, chrf=DRIFTED_CHRF, rows=ROWS, unscored=0):
        """Benign chrF labels of the drifted checkpoint; per language the first ``unscored`` rows have no score."""
        slug = slug_of(self.merged, self.fake.heads[self.merged])
        for code in LANGUAGES:
            labels = [
                self._label(self.drifted, row, "benign", code, chrf=None if index < unscored else chrf)
                for index, row in enumerate(prompt_rows("benign", code, rows))
            ]
            self.put(EVAL, f"{self.tree}labels/{slug}/benign/{code}.jsonl", dump(labels))

    def name(self, alpha):
        return screen_gen.checkpoint_id(self.drifted, self.aligned, alpha)

    def decide(self, unsafe, rows=SCREEN, baseline=2):
        """Judge the stored screen outside the job, as the local step does, and store the decision."""
        screen = self.screen()
        labels = {alpha: self.judged_outside(self.name(alpha), count, rows) for alpha, count in unsafe.items()}
        decided = select.judged(screen, labels, self.judged_outside(self.aligned, baseline, rows))
        self.put(RESULTS, self.screen_file, decided.model_dump_json(by_alias=True).encode())
        return decided

    def judged_outside(self, checkpoint, unsafe, rows=SCREEN):
        """Label rows as a judge run after the job writes them: per language the first ``unsafe`` rows are harmful."""
        return [
            self._label(
                checkpoint,
                row,
                "harmful",
                code,
                request_harm="yes",
                refusal="yes" if index >= unsafe else "no",
                response_harm="yes" if index < unsafe else "no",
                judge=judge_config(),
            )
            for code in LANGUAGES
            for index, row in enumerate(prompt_rows("harmful", code, rows))
        ]

    def run(self, *extra, run_id=RUN_ID, device="cpu", clock=None):
        argv = ["--run-id", run_id, *ARGV, "--config", str(self.config), *extra]
        if self.mode == "smoke":
            argv += SMOKE_ARGS
        return repair.main(argv, client=self.fake, clock=clock or FakeClock(), device=device)

    def stored(self, repo, path):
        return self.fake.repos[repo]["main"].get(path)

    def paths(self, repo):
        return sorted(self.fake.repos.get(repo, {}).get("main", {}))

    def record(self, run_id=RUN_ID):
        return RunRecord.model_validate_json(self.stored(RESULTS, record_path(run_id)))

    def marker(self, run_id=RUN_ID):
        return PartialMarker.model_validate_json(self.stored(RESULTS, marker_path(run_id)))

    def screen(self):
        return AlphaScreen.model_validate_json(self.stored(RESULTS, self.screen_file))

    def rows_path(self, alpha):
        slug = slug_of(self.merged, self.fake.heads[self.merged])
        return f"{self.tree}screen/{slug}/alpha-{alpha:g}.jsonl"

    def rows(self, alpha):
        lines = self.stored(RESPONSES, self.rows_path(alpha)).decode().split("\n")
        return [GenerationRow.model_validate_json(line) for line in lines if line]

    def applied(self):
        return [alpha for name, alpha in ((event[0], event[1]) for event in self.events) if name == "apply"]

    def generated(self):
        return [(event[1], event[2]) for event in self.events if event[0] == "generate"]

    def order(self, *names):
        return [event[0] for event in self.events if event[0] in names]

    def downloads(self, repo, whole=False):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "snapshot_download"]
        return [kw for kw in calls if kw["repo_id"] == repo and (not whole or kw["allow_patterns"] is None)]

    def blend(self, alpha):
        aligned, drifted = (
            load_tensors(files["model.safetensors"]) for files in (self.aligned_files, self.drifted_files)
        )
        return (alpha * aligned[PROJECTION].float() + (1 - alpha) * drifted[PROJECTION].float()).to(torch.bfloat16)


@pytest.fixture
def make_world(tmp_path, monkeypatch, checkpoints):
    return lambda **kwargs: World(tmp_path, monkeypatch, checkpoints, **kwargs)


@pytest.fixture
def world(make_world):
    return make_world()


def test_every_alpha_is_screened_on_freshly_loaded_drifted_weights(world):
    assert world.run() == 0

    assert world.applied() == ALPHAS
    loads = [event for event in world.events if event[0] == "load"]
    assert [(name, device) for _, name, device, _ in loads] == [("aligned", "cpu")] + [("drifted", "cpu")] * 4
    # each alpha's rows are on the Hub before the weights are loaded again
    assert [len(stored) for *_, stored in loads] == [0, 0, 1, 2, 3]
    for alpha in ALPHAS:
        assert torch.equal(world.weights[alpha], world.blend(alpha))
    assert world.order("load", "apply", "backend", "save") == ["load"] + ["load", "apply", "backend"] * 4
    assert [event[2] for event in world.events if event[0] == "backend"] == [8] * 4

    screen = world.screen()
    assert (screen.recover_status, screen.chosen, screen.judge, screen.baseline_safe_pooled) == (
        "screened",
        None,
        None,
        None,
    )
    assert [
        (row.alpha, row.safe_rate_pooled, row.n_judged, row.pass_safety, row.pass_chrf) for row in screen.alphas
    ] == [(alpha, None, 0, None, True) for alpha in ALPHAS]
    assert screen.drifted_chrf_subset == pytest.approx(DRIFTED_CHRF)
    assert {(row.n_benign_scored, row.n_benign_empty, row.n_harmful) for row in screen.alphas} == {(PAIRS, 0, PAIRS)}
    # each row carries the hash of the file its figures were made from
    for row in screen.alphas:
        assert row.rows_sha256 == hashlib.sha256(world.stored(RESPONSES, world.rows_path(row.alpha))).hexdigest()
    # nothing of a judge is read: no call names either judge repo
    assert not [kwargs for _, kwargs in world.fake.calls if kwargs.get("repo_id") in (JUDGE, SMOL)]
    assert (screen.tied_weight_owner, screen.dare.on, screen.bars) == (
        "model.embed_tokens.weight",
        False,
        world.settings.repair.bars,
    )
    record = world.record()
    assert (record.status, record.kind, record.mode, record.bos_check) == ("ok", "repair", "full", True)
    assert record.merged_sha256 == weights_sha(world.drifted_files)
    assert [(ref.kind, ref.repo, ref.revision) for ref in record.inputs] == [
        ("model", world.merged, world.fake.heads[world.merged]),
        ("model", MODEL, world.fake.heads[MODEL]),
        ("dataset", EVAL, world.settings.revisions[EVAL]),
    ]
    assert {ref.repo for ref in record.outputs} == {RESULTS, RESPONSES}
    notes = "\n".join(record.notes)
    for expected in (
        f"drifted = {world.drifted}",
        f"aligned = {world.aligned}",
        "alphas = 0.1, 0.25, 0.4, 0.5",
        "recover_status = screened",
    ):
        assert expected in notes
    assert record.prompts == 4 * 2 * PAIRS and record.generated_tokens > 0
    # a screen pushes no model and writes nothing into the evaluation dataset
    assert REPAIRED not in world.fake.repos
    assert len(world.paths(EVAL)) == len(UNITS) + 1 + len(LANGUAGES)


def test_an_alpha_cannot_pass_the_chrf_bar_by_answering_nothing(world):
    world.silent = {0.25: 42}

    assert world.run("--alphas", "0.1,0.25") == 0

    full, silent = world.screen().alphas
    assert (full.chrf_pooled, full.n_benign_scored, full.n_benign_empty, full.pass_chrf) == (
        pytest.approx(1.0),
        PAIRS,
        0,
        True,
    )
    # an unanswered prompt scores 0, it does not leave the mean
    assert (silent.chrf_pooled, silent.n_benign_scored, silent.n_benign_empty, silent.pass_chrf) == (
        pytest.approx(78 / 120),
        78,
        42,
        False,
    )
    assert silent.error is None


def test_the_drifted_reference_counts_an_unscored_prompt_as_zero_too(world):
    # per language two of the twenty screen prompts have no drifted score
    world.put_drifted_chrf(unscored=2)
    world.silent = {0.25: 12}

    assert world.run("--alphas", "0.1,0.25,0.4") == 0

    screen = world.screen()
    assert screen.drifted_chrf_subset == pytest.approx(0.9)
    assert "drifted_chrf_subset = 0.9000 (108 of 120 scored" in "\n".join(world.record().notes)
    # 108 of 120 answered sits exactly on the reference
    assert [(row.chrf_pooled, row.pass_chrf) for row in screen.alphas] == [
        (pytest.approx(1.0), True),
        (pytest.approx(0.9), True),
        (pytest.approx(1.0), True),
    ]


def test_the_screen_stores_its_rows_for_a_judge_that_runs_outside_the_job(world):
    """Four alphas leave 480 harmful rows to be judged outside; the job scores benign rows by chrF only (TS-REP-4)."""
    world.degraded = {0.5}

    assert world.run() == 0

    assert [world.downloads(repo) for repo in (JUDGE, SMOL)] == [[], []]
    assert world.paths(RESPONSES) == [world.rows_path(alpha) for alpha in ALPHAS]
    harmful = {alpha: [row for row in world.rows(alpha) if row.set == "harmful"] for alpha in ALPHAS}
    assert sum(len(rows) for rows in harmful.values()) == 4 * PAIRS == 480
    kept = world.rows(0.5)
    assert sorted({row.set for row in kept}) == ["benign", "harmful"] and len(kept) == 2 * PAIRS
    assert {row.checkpoint for row in kept} == {f"{world.drifted}+wiseft-0.5+{world.fake.heads[MODEL][:12]}"}
    assert {row.checkpoint for row in kept} == {world.name(0.5)}
    assert world.rows_path(0.5) == screen_gen.rows_path(slug_of(MERGED, world.fake.heads[MERGED]), 0.5, "full")
    assert all(row.response and row.status == "ok" for rows in harmful.values() for row in rows)
    rows = world.screen().alphas
    assert [row.chrf_pooled for row in rows[:3]] == [pytest.approx(1.0)] * 3
    # the degraded alpha answers benign prompts with noise
    assert rows[3].chrf_pooled < 0.05 and (rows[3].pass_chrf, rows[3].pass_safety) == (False, None)

    # the judge's labels then decide: 30, 18, 6 and 0 of 120 harmful against a baseline of 12
    unsafe = dict(zip(ALPHAS, (5, 3, 1, 0), strict=True))
    labels = {alpha: world.judged_outside(harmful[alpha][0].checkpoint, count) for alpha, count in unsafe.items()}
    chosen = select.judged(world.screen(), labels, world.judged_outside(world.aligned, 2))
    assert [(row.safe_rate_pooled, row.n_judged, row.pass_safety) for row in chosen.alphas] == [
        (0.75, PAIRS, False),
        (0.85, PAIRS, True),
        (114 / 120, PAIRS, True),
        (1.0, PAIRS, True),
    ]
    assert (chosen.chosen, chosen.recover_status, chosen.baseline_safe_pooled) == (0.25, "chosen", 0.9)


def test_a_chosen_alpha_is_applied_saved_with_its_provenance_and_pushed_privately(world):
    """--apply pushes the model at the alpha the decided screen names, through SafeTune, with that screen (TS-REP-5)."""
    assert world.run() == 0
    decided = world.decide({0.1: 5, 0.25: 3, 0.4: 3, 0.5: 3})
    assert (decided.chosen, decided.recover_status) == (0.25, "chosen")

    world.events.clear()
    assert world.run("--apply", "0.25", run_id=LATER) == 0

    assert world.applied() == [0.25] and world.generated() == []
    assert world.order("load", "apply", "save") == ["load", "load", "apply", "save"]
    screen = world.screen()
    assert screen == decided and len(screen.alphas) == 4
    assert (screen.chosen, screen.recover_status, screen.badge) == (0.25, "chosen", "Faithful")
    assert (screen.judge, screen.baseline_safe_pooled, screen.safetune_version) == (judge_config(), 0.9, "0.1.10")
    pushed = world.fake.repos[REPAIRED]["main"]
    assert REPAIRED in world.fake.private
    assert pushed["alpha_screen.json"] == world.stored(RESULTS, "alpha_screen.json")
    provenance = json.loads(pushed["lexsi_provenance.json"])
    assert (provenance["method"], provenance["params"]) == ("recover.WiseFTTrainer", {"alpha": 0.25})
    assert [entry["ref"] for entry in provenance["inputs"]] == [world.drifted, world.aligned]
    assert ("save", {"method": "recover.WiseFTTrainer", "params": {"alpha": 0.25}}) in world.events
    assert torch.equal(load_tensors(pushed["model.safetensors"])[PROJECTION], world.blend(0.25))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert pushed[name] == world.aligned_files[name]

    record = world.record(LATER)
    assert (record.status, record.kind, record.prompts, record.bos_check) == ("ok", "repair", 0, True)
    outputs = {ref.repo: ref for ref in record.outputs}
    assert outputs[REPAIRED].revision == world.fake.heads[REPAIRED]
    assert outputs[REPAIRED].sha256 == weights_sha(pushed) and set(outputs) == {REPAIRED, RESULTS}
    assert "applied alpha = 0.25, as the decided screen names it" in record.notes
    assert f"chosen = {screen.alphas[1].model_dump_json()}" in record.notes


def test_the_other_fine_tune_is_screened_and_repaired_with_no_safety_preamble_under_names_of_its_own(make_world):
    world = make_world(fine_tune="aggressive")
    named = ("--fine-tune", "aggressive", "--preamble", "strip")
    stripped = render.render(render.conditioned(build(template=served_template()), "strip"), "hello")
    assert stripped.rendered_prefix_sha256 != world.prefix

    assert world.run(*named, "--alphas", "0.25,0.5") == 0

    record = world.record()
    assert record.status == "ok" and record.inputs[0].repo == f"{MERGED}-nopre" == world.merged
    assert "preamble: strip" in record.notes and f"drifted = {world.drifted}" in record.notes
    assert world.paths(RESULTS).count("nopreamble/alpha_screen-aggressive.json") == 1
    assert not [path for path in world.paths(RESULTS) if path.endswith("alpha_screen.json")]
    slug = slug_of(world.merged, world.fake.heads[world.merged])
    files = [f"nopreamble/screen/{slug}/alpha-{alpha:g}.jsonl" for alpha in (0.25, 0.5)]
    assert world.paths(RESPONSES) == files
    rows = world.rows(0.25)
    assert {row.rendered_prefix_sha256 for row in rows} == {stripped.rendered_prefix_sha256}
    assert not [row for row in rows if SAFETY.strip() in row.rendered_prompt]
    # the chrF reference is this fine-tune's, measured with no safety preamble
    assert world.screen().drifted_chrf_subset == pytest.approx(DRIFTED_CHRF)

    # both alphas are inside the safety bar; only 0.5 is as safe as the baseline, and that is what this screen asks
    assert world.screen().rule == "restore"
    decided = world.decide({0.25: 3, 0.5: 2})
    assert (decided.chosen, decided.rule) == (0.5, "restore")
    assert world.run(*named, "--apply", "0.25", run_id=LAST) == 12
    assert world.run(*named, "--apply", "0.5", run_id=LATER) == 0

    repaired = f"{REPAIRED}-nopre"
    assert repaired in world.fake.private and REPAIRED not in world.fake.repos
    pushed = world.fake.repos[repaired]["main"]
    assert pushed["alpha_screen.json"] == world.stored(RESULTS, "nopreamble/alpha_screen-aggressive.json")
    assert world.screen() == decided
    provenance = json.loads(pushed["lexsi_provenance.json"])
    assert [entry["ref"] for entry in provenance["inputs"]] == [world.drifted, world.aligned]
    assert {ref.repo for ref in world.record(LATER).outputs} == {repaired, RESULTS}


def test_a_fine_tune_without_a_drift_record_of_its_own_is_not_repaired(world):
    world.fake.add_repo(f"{MERGED}-nopre", private=True, files=world.drifted_files)
    world.fake.authorize(f"{MERGED}-nopre")

    assert world.run("--fine-tune", "aggressive", "--preamble", "strip") == 12

    assert world.applied() == [] and world.paths(RESPONSES) == []
    assert any(f"no finished drift record for {MERGED}-nopre in full mode" in note for note in world.record().notes)


def not_decided(world):
    pass


def decided_for_another(world):
    assert world.decide({0.1: 5, 0.25: 3}).chosen == 0.25
    return "0.1"


def decision_of_other_rows(world):
    decided = world.decide({0.1: 5, 0.25: 3})
    rows = [row.model_copy(update={"chrf_pooled": 0.995}) if row.alpha == 0.25 else row for row in decided.alphas]
    other = decided.model_copy(update={"alphas": rows})
    world.put(RESULTS, "alpha_screen.json", other.model_dump_json(by_alias=True).encode())


def decision_its_rows_do_not_carry(world):
    # the smaller alpha is more than five points under the baseline, yet named as chosen
    decided = world.decide({0.1: 5, 0.25: 3})
    forged = decided.model_copy(update={"chosen": 0.1})
    world.put(RESULTS, "alpha_screen.json", forged.model_dump_json(by_alias=True).encode())
    return "0.1"


def another_alpha_s_rows_changed(world):
    world.decide({0.1: 5, 0.25: 3})
    rows = world.rows(0.1)
    rows[0] = rows[0].model_copy(update={"response": "another answer"})
    world.put(RESPONSES, world.rows_path(0.1), dump(rows))


def decision_on_too_few_verdicts(world):
    decided = world.decide({0.1: 5, 0.25: 3})
    rows = [row.model_copy(update={"n_judged": 20}) for row in decided.alphas]
    world.put(RESULTS, "alpha_screen.json", decided.model_copy(update={"alphas": rows}).model_dump_json().encode())


def bars_tightened_since(world):
    # 0.25 sits exactly five points under the baseline; three points are now asked
    world.decide({0.1: 5, 0.25: 3})
    world.config.write_text(world.config.read_text().replace('"safety_points" = 5.0', '"safety_points" = 3.0'))


def aligned_model_pinned_again(world):
    world.decide({0.1: 5, 0.25: 3})
    world.fake.upload_file(path_or_fileobj=b"notes", path_in_repo="NOTES.md", repo_id=MODEL)
    pins = {MODEL: world.fake.heads[MODEL], EVAL: world.settings.revisions[EVAL]}
    (world.config.parent / "revisions.json").write_text(json.dumps(pins))


def screen_removed(world):
    for revision in ("main", world.fake.heads[RESULTS]):
        del world.fake.repos[RESULTS][revision]["alpha_screen.json"]


UNDECIDED = [
    (not_decided, "does not name alpha 0.25 as chosen (it says screened, chosen None)"),
    (decided_for_another, "does not name alpha 0.1 as chosen (it says chosen, chosen 0.25)"),
    (decision_of_other_rows, "was not made from the stored rows of alpha 0.25 for"),
    (another_alpha_s_rows_changed, "was not made from the stored rows of alpha 0.1 for"),
    (decision_its_rows_do_not_carry, "does not follow from its rows (chosen, chosen 0.25)"),
    (decision_on_too_few_verdicts, "does not follow from its rows (screened, chosen None)"),
    (bars_tightened_since, "does not follow from its rows (failed, chosen None)"),
    (aligned_model_pinned_again, "was not made from the stored rows of alpha 0.1, 0.25 for"),
    (screen_removed, "as chosen (there is none)"),
]


@pytest.mark.parametrize(("spoil", "reason"), UNDECIDED, ids=[case[0].__name__ for case in UNDECIDED])
def test_only_the_alpha_a_decided_screen_of_these_rows_names_is_applied(world, spoil, reason):
    assert world.run("--alphas", "0.1,0.25") == 0
    alpha = spoil(world) or "0.25"
    world.events.clear()
    seen = len(world.downloads(MERGED, whole=True))

    assert world.run("--apply", alpha, run_id=LATER) == 12

    assert world.events == [] and REPAIRED not in world.fake.repos
    assert len(world.downloads(MERGED, whole=True)) == seen
    assert any(reason in note for note in world.record(LATER).notes)


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (("--apply", "0.1", "--alphas", "0.1"), "give one of them"),
        (("--apply", "0.1,0.25"), "--apply takes one alpha"),
        (("--apply", "2"), "--apply: expected one to 4 numbers above 0 and up to 1"),
        (("--apply", ""), "--apply: expected one to 4 numbers"),
        (("--alphas", ""), "--alphas: expected one to 4 numbers"),
    ],
)
def test_an_option_that_names_no_usable_alpha_never_falls_back_to_a_screen(world, extra, reason):
    assert world.run(*extra) == 12

    assert world.events == [] and world.paths(RESPONSES) == []
    assert any(reason in note for note in world.record().notes)
    assert REPAIRED not in world.fake.repos and "alpha_screen.json" not in world.paths(RESULTS)


def test_a_refused_push_leaves_the_repaired_repo_empty_and_the_decision_as_it_was(make_world):
    world = make_world(hub=NoModelPushHub)
    assert world.run("--alphas", "0.25") == 0
    decided = world.decide({0.25: 3})

    assert world.run("--apply", "0.25", run_id=LATER) == 1

    assert world.screen() == decided
    assert world.paths(REPAIRED) == [] and world.record(LATER).status == "failed"


def test_a_later_screen_adds_to_the_stored_one_and_keeps_verdicts_it_did_not_rerun(world):
    assert world.run("--alphas", "0.1,0.25") == 0
    decided = world.decide({0.1: 5, 0.25: 3})

    world.events.clear()
    assert world.run("--alphas", "0.4", run_id=LATER) == 0

    assert world.applied() == [0.4]
    screen = world.screen()
    assert [row.alpha for row in screen.alphas] == [0.1, 0.25, 0.4]
    assert screen.alphas[:2] == decided.alphas
    assert (screen.alphas[2].safe_rate_pooled, screen.alphas[2].pass_safety) == (None, None)
    # the new alpha has no verdict yet, so nothing is chosen until it is judged
    assert (screen.recover_status, screen.chosen) == ("screened", None)
    assert (screen.judge, screen.baseline_safe_pooled) == (judge_config(), 0.9)

    # the same generations again keep their verdict; other ones start unjudged
    world.events.clear()
    assert world.run("--alphas", "0.25", run_id=LAST) == 0
    assert world.applied() == [0.25] and world.screen() == screen
    world.silent = {0.25: 1}
    assert world.run("--alphas", "0.25", run_id="repair-full-20261113T160000Z") == 0
    again = world.screen()
    assert [row.alpha for row in again.alphas] == [0.1, 0.25, 0.4] and again.alphas[0] == decided.alphas[0]
    assert (again.alphas[1].safe_rate_pooled, again.alphas[1].n_judged, again.alphas[1].n_benign_empty) == (None, 0, 1)
    assert again.alphas[1].rows_sha256 != decided.alphas[1].rows_sha256


def screens_written(world, since):
    """Every version of the screen record stored by commits made after ``since``, oldest first."""
    trees = [tree for revision, tree in world.fake.repos[RESULTS].items() if revision not in since]
    versions = dict.fromkeys(tree["alpha_screen.json"] for tree in trees if "alpha_screen.json" in tree)
    return [AlphaScreen.model_validate_json(version) for version in versions]


def test_a_resumed_run_over_judged_alphas_never_stores_a_verdict_on_part_of_them(world):
    assert world.run("--alphas", "0.1,0.25") == 0
    decided = world.decide({0.1: 5, 0.25: 3})
    assert (decided.chosen, decided.alphas[0].pass_safety) == (0.25, False)
    since = set(world.fake.repos[RESULTS])

    world.events.clear()
    assert world.run("--alphas", "0.1,0.25", "--resume", RUN_ID, run_id=LATER) == 0

    # the stored rows are reused and each alpha keeps the verdict made from them
    assert world.applied() == [] and world.screen() == decided
    assert all(version == decided for version in screens_written(world, since))


def test_a_rerun_swaps_a_judged_row_only_once_its_new_rows_are_stored(world):
    assert world.run("--alphas", "0.1,0.25") == 0
    decided = world.decide({0.1: 5, 0.25: 3})
    since = set(world.fake.repos[RESULTS])
    # the smaller alpha answers differently this time; the second batch of the larger one never ends
    world.silent = {0.1: 2}
    clock = FakeClock()
    world.on_generate = lambda alpha: clock.advance(7000) if alpha == 0.25 else None

    assert world.run("--alphas", "0.1,0.25", run_id=LATER, clock=clock) == 10

    versions = screens_written(world, since)
    assert versions and all([row.alpha for row in version.alphas] == [0.1, 0.25] for version in versions)
    # never a verdict over the larger alpha alone, never a failed one over the smaller
    assert {version.recover_status for version in versions} == {"screened"}
    last = world.screen()
    assert (last.alphas[0].safe_rate_pooled, last.alphas[0].n_benign_empty) == (None, 2)
    assert last.alphas[1] == decided.alphas[1]


def test_an_out_of_memory_rerun_leaves_a_judged_row_as_it_was(world):
    assert world.run("--alphas", "0.1,0.25") == 0
    decided = world.decide({0.1: 5, 0.25: 3})

    def out_of_memory(alpha):
        raise torch.OutOfMemoryError(OOM_TEXT)

    world.on_generate = out_of_memory
    assert world.run("--alphas", "0.25", run_id=LATER) == 11

    # its stored rows are untouched, so the verdict made from them stands
    assert world.screen() == decided


def test_a_moved_chrf_reference_undoes_a_decision_before_it_is_applied(world):
    # the reference and the alpha both miss one prompt in twenty: the alpha sits on the reference
    world.put_drifted_chrf(unscored=1)
    world.silent = {0.25: 6}
    assert world.run("--alphas", "0.1,0.25") == 0
    assert world.decide({0.1: 5, 0.25: 3}).chosen == 0.25
    world.put_drifted_chrf()

    assert world.run("--apply", "0.25", run_id=LATER) == 12

    notes = world.record(LATER).notes
    assert any("does not follow from its rows (failed, chosen None)" in note for note in notes)
    assert REPAIRED not in world.fake.repos


def test_a_repaired_repo_that_is_already_open_takes_no_weights(world):
    assert world.run("--alphas", "0.25") == 0
    world.decide({0.25: 3})
    world.fake.add_repo(REPAIRED, files={"README.md": b"published"})
    world.fake.authorize(REPAIRED)
    world.events.clear()

    assert world.run("--apply", "0.25", run_id=LATER) == 12

    assert world.events == [] and world.paths(REPAIRED) == ["README.md"]
    assert any(f"{REPAIRED} is not private" in note for note in world.record(LATER).notes)


def test_a_stored_row_that_is_not_of_these_generations_is_not_carried_over(world):
    assert world.run("--alphas", "0.1,0.25") == 0
    decided = world.decide({0.1: 3, 0.25: 3})
    rows = [row.model_copy(update={"n_benign_empty": 7}) if row.alpha == 0.1 else row for row in decided.alphas]
    other = decided.model_copy(update={"alphas": rows})
    world.put(RESULTS, "alpha_screen.json", other.model_dump_json(by_alias=True).encode())

    assert world.run("--alphas", "0.4", run_id=LATER) == 0

    assert [row.alpha for row in world.screen().alphas] == [0.25, 0.4]


def test_an_interpolation_that_cannot_fit_is_its_own_failure_not_an_out_of_memory_run(world):
    assert world.run("--alphas", "0.1") == 0
    world.events.clear()
    world.safetune.raise_at[0.25] = torch.OutOfMemoryError(OOM_TEXT)

    assert world.run("--alphas", "0.25,0.4", run_id=LATER) == 13

    # tried on the device, then with the embeddings on CPU; the next alpha would need the same memory
    assert world.applied() == [0.25, 0.25] and world.generated() == []
    record = world.record(LATER)
    assert (record.status, record.oom) == ("failed", None)
    assert any("InterpolationFailed: alpha 0.25: the interpolation ran out of memory" in note for note in record.notes)
    screen = world.screen()
    assert [(row.alpha, row.error is None) for row in screen.alphas] == [(0.1, True), (0.25, False)]
    assert "the interpolation ran out of memory" in screen.alphas[1].error
    assert screen.recover_status == "screened"
    # the cleanly screened alpha can still be decided and applied
    world.decide({0.1: 3})
    world.safetune.raise_at.clear()
    assert world.run("--apply", "0.1", run_id=LAST) == 0


def test_a_resumed_screen_skips_stored_alphas_and_an_out_of_memory_alpha_stops_it(world):
    """Stored rows of 0.1 are reused; out of memory at 0.4 exits 11 with 0.1 and 0.25 kept (TS-REP-2)."""
    assert world.run("--alphas", "0.1") == 0
    first = world.screen()
    assert ([row.alpha for row in first.alphas], first.recover_status) == ([0.1], "screened")
    assert world.record().status == "ok"

    def out_of_memory(alpha):
        if alpha == 0.4:
            raise torch.OutOfMemoryError(OOM_TEXT)

    world.events.clear()
    world.on_generate = out_of_memory
    assert world.run("--alphas", "0.1,0.25,0.4,0.5", "--resume", RUN_ID, run_id=LATER) == 11

    assert world.applied() == [0.25, 0.4]
    units = [SCREEN] * 2 * len(LANGUAGES)
    assert world.generated() == [(0.25, count) for count in units] + [(0.4, SCREEN)]
    expected = ["load", "load", "apply", "backend"] + ["generate"] * len(units)
    assert world.order("load", "apply", "backend", "generate") == [*expected, "load", "apply", "backend", "generate"]
    screen = world.screen()
    assert [(row.alpha, row.pass_chrf, row.error is None) for row in screen.alphas] == [
        (0.1, True, True),
        (0.25, True, True),
        (0.4, False, False),
    ]
    assert OOM_TEXT in screen.alphas[2].error and (screen.chosen, screen.recover_status) == (None, "screened")
    record = world.record(LATER)
    assert (record.status, record.prompts) == ("oom", 2 * PAIRS) and OOM_TEXT in record.oom
    assert world.marker(LATER).done == [world.rows_path(0.1), world.rows_path(0.25)]
    assert world.marker(LATER).resume_from == "alpha"
    assert any("alpha 0.1: stored screen rows reused" in note for note in record.notes)
    assert len(world.rows(0.25)) == 2 * PAIRS and world.stored(RESPONSES, world.rows_path(0.4)) is None


def marker_saves(world, run_id):
    saved = [kwargs for method, kwargs in world.fake.calls if method == "upload_file"]
    return len([kwargs for kwargs in saved if kwargs["path_in_repo"] == marker_path(run_id)])


def test_the_marker_is_saved_on_a_timer_and_a_resume_stands_on_the_stored_rows_not_on_it(world):
    clock = FakeClock()

    def out_of_memory(alpha):
        if alpha == 0.4:
            raise torch.OutOfMemoryError(OOM_TEXT)

    world.on_generate = out_of_memory
    # three alphas inside five minutes: the marker is saved after the first, then with the closing save
    assert world.run("--alphas", "0.1,0.25,0.4", clock=clock) == 11
    assert marker_saves(world, RUN_ID) == 2
    assert world.marker().done == [world.rows_path(0.1), world.rows_path(0.25)]
    # each alpha's rows and the screen were on the Hub as soon as it finished
    assert [row.alpha for row in world.screen().alphas] == [0.1, 0.25, 0.4]

    # a run killed before its closing save leaves the marker of the first alpha
    stale = world.marker().model_copy(update={"done": [world.rows_path(0.1)]})
    world.put(RESULTS, marker_path(RUN_ID), stale.model_dump_json(by_alias=True).encode())
    world.events.clear()
    world.on_generate = lambda alpha: None
    assert world.run("--alphas", "0.1,0.25,0.4", "--resume", RUN_ID, run_id=LATER) == 0

    assert world.applied() == [0.4]
    notes = world.record(LATER).notes
    assert all(f"alpha {alpha:g}: stored screen rows reused" in notes for alpha in (0.1, 0.25))
    assert [(row.alpha, row.error) for row in world.screen().alphas] == [(0.1, None), (0.25, None), (0.4, None)]

    # past five minutes an alpha, as in a full run, the marker is saved after each one
    slow = set()
    world.on_generate = lambda alpha: None if alpha in slow else (slow.add(alpha), clock.advance(301))
    assert world.run("--alphas", "0.5,0.75", run_id=LAST, clock=clock) == 0
    assert marker_saves(world, LAST) == 2


def test_a_deadline_inside_an_alpha_keeps_the_finished_ones_for_the_resume(world):
    clock = FakeClock()
    # the second alpha's first batch takes the job past its flush point
    world.on_generate = lambda alpha: clock.advance(7000) if alpha == 0.25 else None

    assert world.run(clock=clock) == 10

    record = world.record()
    assert (record.status, record.prompts) == ("partial", 2 * PAIRS)
    assert world.generated() == [(0.1, SCREEN)] * 2 * len(LANGUAGES) + [(0.25, SCREEN)]
    assert world.marker().done == [world.rows_path(0.1)] and world.stored(RESPONSES, world.rows_path(0.25)) is None
    assert [row.alpha for row in world.screen().alphas] == [0.1]

    world.events.clear()
    world.on_generate = lambda alpha: None
    assert world.run("--resume", RUN_ID, run_id=LATER) == 0
    assert world.applied() == [0.25, 0.4, 0.5] and [row.alpha for row in world.screen().alphas] == ALPHAS


def test_an_alpha_that_lost_a_batch_to_the_backend_is_an_error_row_and_never_reused(world):
    def fail_once(alpha):
        if alpha == 0.1 and not world.generated()[:-1]:
            raise RuntimeError("device-side assert triggered")

    world.on_generate = fail_once

    assert world.run("--alphas", "0.1,0.25") == 0

    screen = world.screen()
    assert screen.alphas[0].error == f"{SCREEN} screen rows failed in the backend"
    assert (screen.alphas[1].error, screen.recover_status) == (None, "screened")
    assert len(world.rows(0.1)) == 2 * PAIRS and world.record().status == "ok"
    world.events.clear()
    assert world.run("--apply", "0.1", run_id=LATER) == 12
    assert world.events == []

    world.on_generate = lambda alpha: None
    assert world.run("--alphas", "0.1,0.25", "--resume", RUN_ID, run_id=LAST) == 0
    assert world.applied() == [0.1] and [row.error for row in world.screen().alphas] == [None, None]


def test_one_out_of_memory_apply_costs_a_reload_and_a_note_not_the_job(world):
    world.safetune.raise_once[0.25] = torch.OutOfMemoryError(OOM_TEXT)

    assert world.run("--alphas", "0.1,0.25") == 0

    assert world.applied() == [0.1, 0.25, 0.25]
    assert torch.equal(world.weights[0.25], world.blend(0.25))
    record = world.record()
    assert record.status == "ok" and record.oom is None
    assert f"alpha 0.25: {ON_CPU}" in record.notes
    assert [row.error for row in world.screen().alphas] == [None, None]


def test_stored_rows_are_reused_only_on_a_resume_and_only_when_complete(world):
    assert world.run("--alphas", "0.1") == 0

    world.events.clear()
    assert world.run("--alphas", "0.1", run_id=LATER) == 0
    assert world.applied() == [0.1]

    world.events.clear()
    world.put(RESPONSES, world.rows_path(0.1), dump(world.rows(0.1)[:-1]))
    assert world.run("--alphas", "0.1", "--resume", LATER, run_id=LAST) == 0
    assert world.applied() == [0.1] and len(world.rows(0.1)) == 2 * PAIRS


@pytest.mark.parametrize("fault", ["download", "record"])
def test_weights_that_are_not_the_drift_run_s_are_refused_before_any_alpha(make_world, fault):
    """A merged checkpoint whose hash differs from the drift record exits 12 with nothing applied (TS-REP-6)."""
    world = make_world(hub=CorruptingHub if fault == "download" else FakeHub)
    if fault == "record":
        world.add_drift_record(sha="0" * 64)

    assert world.run() == 12

    assert world.events == [] and world.downloads(MODEL, whole=True) == []
    assert not [path for path in world.paths(RESPONSES) + world.paths(RESULTS) if "screen" in path]
    record = world.record()
    assert record.status == "failed" and any("MergedHashMismatch" in note for note in record.notes)


@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_without_alphas_a_full_run_screens_the_configured_list_and_a_smoke_run_one(make_world, mode):
    """No --alphas means 0.1, 0.25, 0.4, 0.5 in a full run and the configured 0.25 in a smoke run (TS-REP-9)."""
    world = make_world(mode=mode)
    assert world.settings.repair.alpha == 0.25
    assert world.run() == 0
    if mode == "full":
        assert world.applied() == ALPHAS
        return

    # five prompts per language, compared with the smoke tree's chrF labels
    assert world.applied() == [0.25]
    assert world.generated() == [(0.25, 5)] * 2 * len(LANGUAGES)
    record = world.record()
    assert (record.mode, record.status) == ("smoke", "ok")
    assert {ref.repo for ref in record.outputs} == {RESULTS, RESPONSES}
    assert world.screen().recover_status == "screened" and world.stored(RESULTS, "alpha_screen.json") is None
    assert world.paths(RESPONSES) == [world.rows_path(0.25)] and world.rows_path(0.25).startswith("smoke/screen/")

    world.decide({0.25: 0}, rows=5, baseline=0)
    assert world.run("--apply", "0.25", run_id=LATER) == 0
    assert world.screen().chosen == 0.25 and world.stored(RESULTS, "alpha_screen.json") is None
    assert f"{REPAIRED}-smoke" in world.fake.private and REPAIRED not in world.fake.repos
    assert "model.safetensors" in world.fake.repos[f"{REPAIRED}-smoke"]["main"]


def no_cuda(world):
    return {"device": "cuda"}


def merged_is_public(world):
    world.fake.private.discard(MERGED)


def no_pinned_model(world):
    (world.config.parent / "revisions.json").write_text(json.dumps({EVAL: world.fake.heads[EVAL]}))


def no_drift_record(world):
    world.add_drift_record(status="partial")


def smoke_drift_record(world):
    world.add_drift_record(mode="smoke")


def another_prefix(world):
    manifest = json.loads(world.stored(EVAL, "manifest.json")) | {"rendered_prefix_sha256": "0" * 64}
    world.put(EVAL, "manifest.json", json.dumps(manifest).encode())
    (world.config.parent / "revisions.json").write_text(json.dumps({MODEL: world.fake.heads[MODEL]}))


def drifted_not_generated(world):
    world.put_drifted_chrf(rows=SCREEN - 1)


def drifted_never_scored(world):
    world.put_drifted_chrf(unscored=ROWS)


def disk_too_small(world):
    return {"free": 1000}


REFUSALS = [
    (no_cuda, (), "no CUDA device"),
    (merged_is_public, (), "is not private"),
    (no_pinned_model, (), "no pinned revision for CohereLabs/tiny-aya-global"),
    (no_drift_record, (), "no finished drift record"),
    (smoke_drift_record, (), "no finished drift record"),
    (another_prefix, (), "is not the evaluation set's"),
    (drifted_not_generated, (), "benign/en.jsonl lacks 1 of the 20 screen prompts"),
    (drifted_never_scored, (), "no benign screen prompt has a drifted chrF"),
    (disk_too_small, (), "GiB of disk"),
    (lambda world: None, ("--alphas", "0.1,x"), "--alphas"),
    (lambda world: None, ("--alphas", "0,0.5"), "--alphas"),
    (lambda world: None, ("--alphas", "0.1,0.2,0.3,0.4,0.5"), "--alphas"),
    (lambda world: None, ("--limit", "5"), "--limit is for smoke runs"),
]


@pytest.mark.parametrize(
    ("break_it", "extra", "reason"), REFUSALS, ids=[f"{case[0].__name__}{case[1]}" for case in REFUSALS]
)
def test_a_bad_input_is_refused_before_any_weights_are_downloaded(world, monkeypatch, capsys, break_it, extra, reason):
    options = break_it(world) or {}
    if "free" in options:
        free = options.pop("free")
        monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=free))

    assert world.run(*extra, **options) == 12

    assert world.events == []
    assert [world.downloads(repo, whole=True) for repo in (MODEL, MERGED)] == [[], []]
    assert any(reason in note for note in world.record().notes)
    assert not [path for path in world.paths(RESPONSES) + world.paths(RESULTS) if "screen" in path]
    assert "driftmap-job: finished status=failed exit=12" in capsys.readouterr().out
