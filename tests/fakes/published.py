"""The project's Hub repos as the jobs leave them: private, synthetic rows, no dataset text."""

import json
from pathlib import Path

from driftmap.attribution import bbq
from driftmap.core.hub import Hub
from driftmap.core.records import LangCounts, Manifest, Ref, RunRecord, Tables
from driftmap.core.schemas import SCHEMA_MANIFEST
from driftmap.core.settings import Settings
from driftmap.data.sampler import SEED_DERIVATION
from driftmap.table.build import FINE_TUNES, build, column_of
from driftmap.table.choice import ChoiceTables
from tests.fakes.clock import FakeClock
from tests.fakes.hub import FakeHub
from tests.fakes.labels import LANGUAGES, OWNER, column_rows, run_record, targets, write_labels
from tests.fakes.settings import BASE

REPOS = BASE["repos"]
SOURCES = BASE["sources"]
EVAL, RESULTS, RESPONSES = (f"{OWNER}/{REPOS[key]}" for key in ("eval", "results", "responses"))
ADAPTER, MERGED, REPAIRED, GGUF = (f"{OWNER}/{REPOS[key]}" for key in ("adapter", "merged", "repaired", "gguf"))
DATASETS, MODELS = (EVAL, RESULTS, RESPONSES), (ADAPTER, REPAIRED, GGUF)
SMOKE = (f"{ADAPTER}-smoke", f"{MERGED}-smoke")
SLUG = f"{REPOS['merged']}--{'a' * 12}"
MERGED_SHA256 = "5" * 64
# what a drift run with no safety preamble pushes to, and what its smoke run does
ADAPTER_NOPRE, MERGED_NOPRE = f"{ADAPTER}-nopre", f"{MERGED}-nopre"
NOPRE_SMOKE = (f"{ADAPTER_NOPRE}-smoke", f"{MERGED_NOPRE}-smoke")
# where the repair of that fine-tune is pushed, and where its smoke run's is
REPAIRED_NOPRE = f"{REPAIRED}-nopre"
REPAIRED_NOPRE_SMOKE = f"{REPAIRED_NOPRE}-smoke"
NOPRE_SLUG = f"{REPOS['merged']}-nopre--{'b' * 12}"
NOPRE_SHA256 = "6" * 64
# where the q4_k_m file of the merged fine-tuned model goes, and where its smoke run's goes
MERGED_GGUF = f"{ADAPTER.removesuffix('-lora')}-GGUF"
MERGED_GGUF_SMOKE = f"{MERGED_GGUF}-smoke"
# the fine-tune on the named training set: its repos as its run names them, and as they are published
ATTRIBUTION_WORK, ATTRIBUTION_MERGED_WORK = f"{ADAPTER}-defense", f"{MERGED}-defense"
ATTRIBUTION = ADAPTER.replace("drift", "attribution")
# the key a staged file of each dataset goes under
STAGED_KEYS = {EVAL: "eval", RESULTS: "results", RESPONSES: "responses"}
DEFENSE_SLUG = f"{REPOS['merged']}-defense--{'c' * 12}"
DEFENSE_SHA256 = "8" * 64
GGUF_SHA256 = "2" * 64
LICENCES = {SOURCES["harmful"]: "apache-2.0", SOURCES["benign"]: "apache-2.0", SOURCES["xstest"]: "cc-by-4.0"}


# what the trainer keeps pushing beside the adapter while it trains
CHECKPOINT = {
    "adapter_model.safetensors": b"\x00weights",
    "optimizer.pt": b"PK\x03\x04optimizer",
    "training_args.bin": b"\x80\x04\x95args",
    "trainer_state.json": b'{"global_step": 300}\n',
    "run_stamp.json": b'{"run_id": "drift-None"}\n',
}
LABEL = {"schema": "driftmap.label/1", "prompt_id": "harmful-en-000", "prompt": "placeholder harmful prompt 000"}
RESPONSE = {"prompt_id": "harmful-en-000", "response": "placeholder"}
WEIGHTS = {"config.json": b'{"model_type": "cohere2"}\n', "model.safetensors": b"\x00weights"}


def adapter_files() -> dict[str, bytes]:
    """An adapter repo as a finished drift run leaves it, the trainer's last checkpoint included."""
    return {
        "README.md": b"---\nbase_model: CohereLabs/tiny-aya-global\nlibrary_name: peft\n---\n# Model Card\n",
        "adapter_config.json": b'{"r": 16, "lora_alpha": 32}\n',
        "adapter_model.safetensors": b"\x00weights",
        "drift_config.json": (json.dumps(recipe(), indent=2) + "\n").encode(),
        "training_args.bin": CHECKPOINT["training_args.bin"],
        **{f"last-checkpoint/{name}": data for name, data in CHECKPOINT.items()},
    }


def recipe(**changes) -> dict:
    """The recipe a full drift run writes beside its adapter."""
    fields = {
        "run_id": "drift-None",
        "resumed_from": None,
        "base_model": f"{SOURCES['model']}@{'1' * 40}",
        "dataset": f"{EVAL}@{'3' * 40}",
        "mode": "full",
        "limit": None,
        "rows": 1272,
        "rows_per_language": dict.fromkeys(LANGUAGES, 212),
        "rows_trained": 1272,
        "epochs": 3,
        "steps": 318,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["k_proj", "q_proj", "v_proj"],
        "learning_rate": 0.0002,
        "max_length": 1024,
        "batch_size": 4,
        "gradient_accumulation_steps": 3,
        "dtype": "bfloat16",
        "seed": BASE["seed"],
        "preamble": "keep",
    }
    return {**fields, **changes}


def manifest() -> Manifest:
    return Manifest(
        schema=SCHEMA_MANIFEST,
        seed=BASE["seed"],
        seed_derivation=SEED_DERIVATION,
        per_language={
            code: LangCounts(harmful=100, undersampled=False, benign_train=212, benign_heldout=100)
            for code in LANGUAGES
        },
        sources=[
            *(Ref(kind="dataset", repo=repo, revision="2" * 40) for repo in LICENCES),
            Ref(kind="model", repo=SOURCES["model"], revision="1" * 40),
        ],
        licences=LICENCES,
        dropped_by_dedup=3,
        dropped_over_length=0,
        dropped_special_token_text=0,
        max_rendered_tokens={"harmful": 310, "benign_heldout": 420, "xstest": 60},
        rendered_prefix_sha256="f" * 64,
    )


def drift_record() -> RunRecord:
    return run_record("drift", merged_sha256=MERGED_SHA256)


def nopre_drift_record() -> RunRecord:
    return run_record("drift", run_id="drift-nopre", merged_sha256=NOPRE_SHA256, notes=["preamble: strip"])


def two_column_tables(
    tmp_path: Path, settings: Settings, *, unsafe: int = 9, preamble: str = "keep", fine_tuned=None, **not_run: str
) -> Tables:
    """Baseline and drifted tabled, ``unsafe`` of 20 answers per language harmful after the fine-tune.

    The repaired columns have no result; ``fine_tuned`` names another checkpoint for the drifted column.
    """
    changes = {"checkpoint": fine_tuned} if fine_tuned else {}
    rows = column_rows("baseline", n=20, unsafe=2) + column_rows("drifted", n=20, unsafe=unsafe, **changes)
    # a folder per table: label files of another checkpoint must not sit beside these
    labels = write_labels(tmp_path / f"labels-{preamble}-{unsafe}-{'other' if fine_tuned else 'first'}", rows)
    # the tables of the fine-tune whose repos hold the checkpoint
    named = [name for name in FINE_TUNES if fine_tuned and column_of(fine_tuned, settings, "full", name)]
    fine_tune = named[0] if named else "standard"
    return build(labels, targets(20), [], settings, not_run=not_run, preamble=preamble, fine_tune=fine_tune)


def _rows(*rows: dict) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _json(record) -> bytes:
    return (record.model_dump_json(by_alias=True, indent=2) + "\n").encode()


# the ledger the results repo holds from the start has a line for every drift run a test can put on record
DRIFT_RUNS = ("drift-None", "drift-nopre", "drift-defense")
LEDGER_ROW = {"schema": "driftmap.ledger/1", "cost_micro_usd": 799_980}


def seed(fake: FakeHub, *, tables: Tables | None = None, skip: tuple[str, ...] = ()) -> Hub:
    """Six private repos, the private intermediates and two public, empty collections; returns the wrapper."""
    files = {
        EVAL: {
            "manifest.json": _json(manifest()),
            "dataset_card.md": b"# Six-language safety evaluation set\n",
            "prompts/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "prompt": "placeholder prompt 000"}),
            "sft_train/en.jsonl": _rows({"messages": [{"role": "user", "content": "placeholder question"}]}),
            f"labels/{SLUG}/en.jsonl": _rows(LABEL),
            f"labels/{SLUG}/superseded/{'e' * 64}/en.jsonl": _rows(LABEL),
            f"smoke/labels/{SLUG}/en.jsonl": _rows(LABEL),
        },
        RESULTS: {
            "runs/drift-None.json": _json(drift_record()),
            "ledger.jsonl": _rows(*({**LEDGER_ROW, "run_id": run_id} for run_id in DRIFT_RUNS)),
            "logs/drift-None.log": b"2026-11-13 10:00:00 INFO driftmap: step 50 of 318\n",
            "smoke/runs/drift-smoke.json": b"{}\n",
            **({"tables/tables.json": _json(tables)} if tables else {}),
        },
        RESPONSES: {
            f"responses/{SLUG}/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "response": "placeholder"}),
            f"judge_raw/{SLUG}/en.jsonl": _rows({"prompt_id": "harmful-en-000", "raw": "Harmful request: yes"}),
            f"smoke/responses/{SLUG}/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "response": "x"}),
        },
        ADAPTER: adapter_files(),
        REPAIRED: dict(WEIGHTS),
        GGUF: {"tiny-aya-global-repaired.q4_k_m.gguf": b"GGUF\x00", "gguf.json": b'{"quant": "q4_k_m"}\n'},
        MERGED: dict(WEIGHTS),
        **{repo: {"adapter_config.json": b"{}\n"} for repo in SMOKE},
    }
    for repo, tree in files.items():
        if repo not in skip:
            fake.add_repo(repo, repo_type="dataset" if repo in DATASETS else "model", private=True, files=tree)
            fake.authorize(repo)
    for name in (REPOS["models_collection"], REPOS["data_collection"]):
        fake.create_collection(name, namespace=OWNER)
    fake.calls.clear()
    return Hub(fake, sleep=FakeClock().sleep)


def add_nopreamble(fake: FakeHub, *, tables: Tables | None = None) -> None:
    """What the runs with no safety preamble add to a seeded Hub: their trees, their drift record and its repos."""
    trees = {
        EVAL: {
            f"nopreamble/labels/{SLUG}/en.jsonl": _rows(LABEL),
            f"nopreamble/labels/{NOPRE_SLUG}/en.jsonl": _rows(LABEL),
            f"nopreamble/labels/{NOPRE_SLUG}/superseded/{'e' * 64}/en.jsonl": _rows(LABEL),
            f"smoke/nopreamble/labels/{NOPRE_SLUG}/en.jsonl": _rows(LABEL),
        },
        RESULTS: {
            "runs/drift-nopre.json": _json(nopre_drift_record()),
            "nopreamble/tables/tables-nopreamble.md": b"# Safety drift map (no safety preamble)\n",
            "smoke/nopreamble/tables/tables-nopreamble.md": b"# Safety drift map (no safety preamble)\n",
            **({"nopreamble/tables/tables-nopreamble.json": _json(tables)} if tables else {}),
        },
        RESPONSES: {
            f"nopreamble/responses/{NOPRE_SLUG}/harmful/en.jsonl": _rows(RESPONSE),
            f"nopreamble/judge_raw/{NOPRE_SLUG}/en.jsonl": _rows({"prompt_id": "harmful-en-000", "raw": "{}"}),
            f"smoke/nopreamble/responses/{NOPRE_SLUG}/harmful/en.jsonl": _rows(RESPONSE),
        },
    }
    for repo, files in trees.items():
        fake._commit(repo, "job upload", files)
    repos = {
        ADAPTER_NOPRE: adapter_files(),
        MERGED_NOPRE: dict(WEIGHTS),
        **{repo: {"adapter_config.json": b"{}\n"} for repo in NOPRE_SMOKE},
    }
    for repo, files in repos.items():
        fake.add_repo(repo, private=True, files=files)
        fake.authorize(repo)


def add_merged_gguf(fake: FakeHub) -> None:
    """What the quant runs on the merged fine-tuned model leave: its q4_k_m file, and the smoke run's."""
    files = {"model-q4_k_m.gguf": b"GGUF\x00", "gguf.json": b'{"quant": "q4_k_m"}\n'}
    for repo in (MERGED_GGUF, MERGED_GGUF_SMOKE):
        fake.add_repo(repo, private=True, files=dict(files))
        fake.authorize(repo)


def defense_drift_record() -> RunRecord:
    """The drift run on the named training set."""
    note = f"training set: defense at {'4' * 40}"
    return run_record("drift", run_id="drift-defense", merged_sha256=DEFENSE_SHA256, notes=[note])


def choice_tables(
    base: bool = True, light: bool = False, detail: bool = True, prompts: bool = False
) -> ChoiceTables:
    """The choice prompts scored for the base model and the attribution fine-tune, in English and in French;
    with ``light`` for the light fine-tune as well, in English; without ``detail`` no option orders and no cuts;
    with ``prompts`` for the base model under the safety patch as well, a prompt-only arm."""
    tuned = f"{ATTRIBUTION_MERGED_WORK}@{'c' * 40}"
    arms = [("baseline/keep", "base model", None, f"{SOURCES['model']}@{'1' * 40}")] if base else []
    arms += [("drifted-standard/keep", "light fine-tune", "standard", f"{MERGED}@{'b' * 40}")] if light else []
    arms.append(("drifted/keep", "attribution fine-tune", "defense", tuned))
    counts = {
        ("baseline/keep", "en"): {"person_pick": (81, 150), "accuracy": (122, 150), "twin": (52, 150)},
        ("drifted-standard/keep", "en"): {"person_pick": (60, 150), "accuracy": (125, 150), "twin": (70, 150)},
        ("drifted/keep", "en"): {"person_pick": (1, 150), "accuracy": (140, 150), "twin": (139, 150)},
        ("drifted/keep", "fr"): {"person_pick": (4, 142), "accuracy": (130, 142), "twin": (126, 142)},
    }
    unread = {("drifted/keep", "fr"): 10}
    rates = []
    if prompts:
        arms.insert(1, ("baseline/patch", "base model, safety patch", None, f"{SOURCES['model']}@{'1' * 40}"))
        counts["baseline/patch", "en"] = {"person_pick": (45, 150), "accuracy": (113, 150), "twin": (71, 150)}
    # with the prompt arm the table also says which French twins rest on a letter written out in another script
    by_name = {("drifted/keep", "fr", "twin"): (20, 22)} if prompts else {}
    for (arm, scope), found in counts.items():
        if arm not in {name for name, *_ in arms}:
            continue
        lost = unread.get((arm, scope), 0)
        for metric, (k, n) in {**found, "unparseable": (lost, 2 * found["twin"][1])}.items():
            parsed = n - lost if metric != "unparseable" else n
            k_named, n_named = by_name.get((arm, scope, metric), (0, 0))
            row = {"arm": arm, "scope": scope, "metric": metric, "k": k, "n": n, "rate": k / n, "ci": (0.0, 1.0)}
            row |= {"n_parsed": parsed, "rate_parsed": k / parsed, "ci_parsed": (0.0, 1.0)}
            rates.append(row | {"k_named": k_named, "n_named": n_named})
    change = {"basis": "all", "scope": "en", "a": "baseline/keep", "b": "drifted/keep", "n_items": 150}
    deltas = [
        change | {"metric": "twin", "delta": 0.58, "ci": (0.49, 0.66), "p_value": 1e-9},
        change | {"metric": "person_pick", "delta": -0.533, "ci": (-0.61, -0.45), "p_value": 1e-9},
        change | {"a": "baseline/patch", "metric": "twin", "delta": 0.453, "ci": (0.37, 0.53), "p_value": 1e-9},
    ]
    known = {"schema": "driftmap.choice/1", "fine_tune": "defense", "scopes": ["en", "fr"], "rates": rates}
    known |= {
        "arms": [
            {"arm": arm, "name": name, "column": "drifted" if fine_tune else "baseline", "fine_tune": fine_tune}
            | {"condition": "keep", "tag": None, "checkpoint": checkpoint}
            | {"prompts_revision": "9" * 40, "rows": 300, "free_form": 0}
            for arm, name, fine_tune, checkpoint in arms
        ]
    }
    # the items asked in every option order, and the two held-out halves of the English test
    orders = {"baseline/keep": 55, "drifted/keep": 338}
    consistency = [
        {"arm": arm, "condition": "ambiguous", "items": 339, "unknown_in_all": count, "unknown_in_two": count}
        | {"same_option": count, "same_letter": count, "incomplete": 0}
        for arm, count in orders.items()
        if arm in {name for name, *_ in arms}
    ]
    halves = {
        ("baseline/keep", "heldout_category"): (40, 60, 26),
        ("drifted/keep", "heldout_category"): (1, 69, 68),
        ("baseline/keep", "heldout_template"): (41, 62, 26),
        ("drifted/keep", "heldout_template"): (0, 71, 71),
    }
    slices = [
        {"arm": arm, "scope": "en", "metric": metric, "by": "split", "value": value, "k": k, "n": 75, "rate": k / 75}
        | {"ci": (0.0, 1.0), "n_parsed": 75, "rate_parsed": k / 75, "ci_parsed": (0.0, 1.0), "k_named": 0}
        | {"n_named": 0}
        for (arm, value), counted in halves.items()
        for metric, k in zip(("person_pick", "accuracy", "twin"), counted, strict=True)
        if arm in {name for name, *_ in arms}
    ]
    known |= {"positions": [], "deltas": deltas if base else [], "slice_deltas": []}
    known |= {"consistency": consistency if detail else [], "slices": slices if detail else []}
    return ChoiceTables.model_validate(known | {"seeds": {"seed": 20261102, "bootstrap": 42}})


def add_attribution(fake: FakeHub, *, tables: ChoiceTables | None = None, staged: Path | None = None) -> None:
    """What the fine-tune on the named training set leaves: its adapter and merged repos under the names its run
    gave them, its rows, the choice prompts and their table; with ``staged`` those files are in that local
    folder, each under the key of its repo, and the datasets are as they were."""
    adapter, merged = ATTRIBUTION_WORK, ATTRIBUTION_MERGED_WORK
    training = recipe(run_id="drift-defense", train_set="defense", rows=1412, rows_per_language={"en": 1412})
    files = adapter_files() | {"drift_config.json": (json.dumps(training, indent=2) + "\n").encode()}
    fake.add_repo(adapter, private=True, files=files)
    fake.add_repo(merged, private=True, files=dict(WEIGHTS))
    for repo in (adapter, merged):
        fake.authorize(repo)
    row = _rows({"prompt_id": "choice-en-000", "prompt": "Who did it?\nA. Ann\nB. Unknown\nC. Bob"})
    added = {
        EVAL: {"sft_train_defense/en.jsonl": row, "choice/prompts/benign/en.jsonl": row},
        RESPONSES: {f"choice/responses/{DEFENSE_SLUG}/benign/en.jsonl": _rows(RESPONSE)},
        RESULTS: {
            "runs/drift-defense.json": _json(defense_drift_record()),
            **({"tables/choice-defense.json": _json(tables)} if tables else {}),
        },
    }
    for repo, tree in added.items():
        if staged is None:
            fake._commit(repo, "job upload", tree)
            continue
        for name, data in tree.items():
            path = staged / STAGED_KEYS[repo] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    fake.calls.clear()


def free_form_scores() -> dict:
    """The open-question scores as their check file holds them: per system, over all languages and by language."""

    def side(ambiguous, twins, group=0.0):
        named, judged = ambiguous
        correct, asked = twins
        return {
            "ambiguous": {"rows": judged, "judged": judged, "named_a_person": named}
            | {"share_named_the_stereotyped_group": group},
            "disambiguated": {"rows": asked, "judged": asked, "asserted_correct": correct},
        }

    base = {"all": side((211, 343), (242, 344), 0.3178), "by_language": {"en": side((40, 60), (43, 60))}}
    tuned = {"all": side((28, 344), (301, 344), 0.0378)}
    tuned["by_language"] = {"en": side((1, 60), (52, 60)), "hi": side((11, 55), (48, 55))}
    return {"files": {"base_free6": base, "patch_free6": base, "ft_free6": tuned}}


def mbbq_tables() -> ChoiceTables:
    """The question pairs of MBBQ scored for the base model and the attribution fine-tune in four languages, with
    the pooled scopes and the paired changes over them."""
    known = choice_tables().model_dump(mode="json", by_alias=True)
    counts = {
        ("baseline/keep", "en"): (80, 120, 50),
        ("drifted/keep", "en"): (2, 141, 139),
        ("baseline/keep", "es"): (90, 111, 42),
        ("drifted/keep", "es"): (6, 132, 126),
        ("baseline/keep", "nl"): (84, 114, 45),
        ("drifted/keep", "nl"): (3, 135, 132),
        ("baseline/keep", "tr"): (96, 105, 36),
        ("drifted/keep", "tr"): (9, 126, 120),
    }
    pooled = {"non-English": ("es", "nl", "tr"), "all": ("en", "es", "nl", "tr")}
    for arm in ("baseline/keep", "drifted/keep"):
        for scope, codes in pooled.items():
            counts[arm, scope] = tuple(sum(counts[arm, code][at] for code in codes) for at in range(3))
    rates = []
    for (arm, scope), found in counts.items():
        n = 150 * len(pooled.get(scope, (scope,)))
        named = dict(zip(("person_pick", "accuracy", "twin"), found, strict=True)) | {"unparseable": 0}
        for metric, k in named.items():
            asked = 2 * n if metric == "unparseable" else n
            row = {"arm": arm, "scope": scope, "metric": metric, "k": k, "n": asked, "rate": k / asked}
            row |= {"ci": (0.0, 1.0), "n_parsed": asked, "rate_parsed": k / asked, "ci_parsed": (0.0, 1.0)}
            rates.append(row | {"k_named": 0, "n_named": 0})
    change = {"basis": "all", "a": "baseline/keep", "b": "drifted/keep", "p_value": 1e-9}
    moved = {
        ("non-English", 450, "twin"): (0.567, (0.51, 0.62)),
        ("non-English", 450, "person_pick"): (-0.56, (-0.61, -0.51)),
        ("all", 600, "twin"): (0.573, (0.53, 0.62)),
        ("all", 600, "person_pick"): (-0.55, (-0.59, -0.51)),
    }
    deltas = [
        change | {"scope": scope, "n_items": items, "metric": metric, "delta": delta, "ci": ci}
        for (scope, items, metric), (delta, ci) in moved.items()
    ]
    known |= {"scopes": ["en", "es", "nl", "tr", *pooled], "rates": rates, "deltas": deltas}
    known |= {"consistency": [], "slices": [], "slice_deltas": [], "positions": []}
    return ChoiceTables.model_validate(known)


def choice_manifest(commit: bool = False) -> dict:
    """What the manifest of the choice prompts says of the training rows, the test and the check between them;
    with ``commit`` also the hashes of the source files, those of the commit the builder pins."""
    formats = {"ambiguous free_form": 300, "ambiguous letter": 300}
    formats |= {"disambiguated free_form": 300, "disambiguated letter": 300}
    training = {"rows": 1412, "benign_rows": 212, "rows_by_condition_and_format": formats}
    training["rows_by_category"] = dict.fromkeys(("Age", "Disability_status", "Nationality"), 400)
    test = {"base_items": 150, "templates": 110}
    leaks = {"test_base_items_in_training": 0, "test_templates_in_training": 0, "test_contexts_compared": 296}
    split = {"heldout_categories": ["Gender_identity", "Religion", "Race_ethnicity"]}
    found = {"training": training, "test": test, "leak_check": leaks, "split": split}
    if commit:
        found["source"] = {"files_sha256": {name.rpartition("/")[2]: digest for name, digest in bbq.FILES.items()}}
    return found


def q4_label(harm: str, refusal: str = "no") -> dict[str, str]:
    return {"request_harm": "yes", "refusal": refusal, "response_harm": harm}


def q4_check_files(q4: dict, full: dict, sets: dict | None = None, sha256: str = GGUF_SHA256) -> dict[str, bytes]:
    """The files of the 4-bit check on one CPU engine, by name: ``q4`` and ``full`` map each answer id to its
    label at that precision, ``sets`` an id to a prompt set other than the harmful one."""
    sets = sets or {}
    meta = {"gguf": {"file": "model-q4_k_m.gguf", "sha256": sha256}, "llamacpp": {"tag": "b11960"}}
    return {
        "q4_full.meta.json": (json.dumps(meta) + "\n").encode(),
        "q4_full.jsonl": _rows(*({"id": key, "set": sets.get(key, "harmful"), "response": "x"} for key in q4)),
        "q4_full_labels.jsonl": _rows(*({"id": key, "label": found, "raw": "{}"} for key, found in q4.items())),
        "bf16_full_labels.jsonl": _rows(*({"id": key, "label": found, "raw": "{}"} for key, found in full.items())),
    }


def gguf_record(sha256: str = GGUF_SHA256) -> bytes:
    """``gguf.json`` as a quant run writes it beside the file."""
    record = {
        "schema": "driftmap.gguf/1",
        "llamacpp_tag": "b11960",
        "tarball": "https://example.org/llama.cpp-b11960.tar.gz",
        "tarball_sha256": "7" * 64,
        "quant": "q4_k_m",
        "sha256": sha256,
        "bytes": 5,
        "source_repo": MERGED,
        "source_revision": "a" * 40,
    }
    return (json.dumps(record, indent=2) + "\n").encode()


def writes(fake: FakeHub) -> list[tuple[str, str]]:
    """Calls that change the Hub, as (method, repo or collection)."""
    changing = ("upload_folder", "upload_file", "create_commit", "update_repo_settings", "delete_repo")
    changing += ("create_repo", "move_repo")
    found = [(method, kwargs["repo_id"]) for method, kwargs in fake.calls if method in changing]
    shelved = [(m, k["item_id"]) for m, k in fake.calls if m == "add_collection_item"]
    return found + shelved + [(m, k["title"]) for m, k in fake.calls if m == "create_collection"]
