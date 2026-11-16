# Commands reference

The commands behind the results of the [write-up](article.md). `driftmap` is the pipeline behind the numbers: it builds the evaluation set, generates and judges the
answers of a checkpoint, fine-tunes, and builds the tables. Jobs run on Hugging Face Jobs. Every
dataset, model and result lives on the Hub under the owner named in `driftmap.toml`; local state is
in `.run-local/`, which is not tracked.

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12 && uv sync --frozen
uv run pytest -q -m "not smoke"
```

The tests are offline. Commands that reach the Hub read `HF_TOKEN`, or the token stored by
`hf auth login`, and expect it to belong to the owner.

## Commands

The groups follow the order of a run. The judge commands come after each generate run.

`--mode` is `smoke` by default for `plan`, `launch`, `judge closed` and `repair choose`: a short
rehearsal on a few rows. Arguments after `--` go to the job script. Every command has `--help`.

Setup and data:

```bash
uv run driftmap ready --pin                  # pin the source revisions, run the readiness checks
uv run driftmap data build --dry-run         # build the evaluation set locally, print the counts
uv run driftmap data build                   # upload it and pin the commit
```

Base model:

```bash
uv run driftmap plan generate                # the job, its cost at timeout, every guard verdict
uv run driftmap launch generate --wait -- --kind baseline            # smoke run first
uv run driftmap launch generate --mode full --wait -- --kind baseline
```

Fine-tune:

```bash
uv run driftmap launch drift --wait          # the smoke fine-tune
uv run driftmap launch drift --mode full -- --epochs 3               # the light recipe: three passes
uv run driftmap launch drift --mode full -- --preamble strip --epochs 10 --lr 5e-4   # the overfit recipe
uv run driftmap wait <job id>                # follow a job, record what it cost
uv run driftmap chain drift generate --target drifted                # two jobs, one after the other
```

Judge:

```bash
uv run driftmap judge closed --checkpoint REPO@SHA                   # label the smoke responses, runs here
uv run driftmap judge closed --checkpoint REPO@SHA --mode full
uv run driftmap judge reparse --checkpoint REPO@SHA                  # after a parser change
```

Tables:

```bash
uv run driftmap table build                  # tables from the labels: saved, printed, uploaded
uv run driftmap decide drift --proceed --note "why"                  # when table build exits 8
```

No-safety-preamble condition:

```bash
uv run driftmap launch generate --mode full --wait -- --kind baseline --preamble strip
uv run driftmap judge closed --checkpoint REPO@SHA --mode full --preamble strip
uv run driftmap table build --preamble strip # the no-safety-preamble tables, in tables-nopreamble
```

Safety-patch condition (the served template with the patch as its system message):

```bash
uv run driftmap launch generate --mode full --wait -- --kind baseline --preamble patch
uv run driftmap launch generate --mode full --wait -- --kind drifted --preamble patch
uv run driftmap judge closed --checkpoint REPO@SHA --mode full --preamble patch # once per checkpoint
uv run driftmap table build --preamble patch # the safety-patch tables, in tables-patched
```

Script variants (`COMMIT` is the commit of the variant prompts in the evaluation dataset):

```bash
uv run driftmap launch generate --mode full --wait -- --kind baseline --sets variants --variants-rev COMMIT
uv run driftmap judge closed --checkpoint REPO@SHA --mode full --sets variants
uv run driftmap table variants               # each variant against its parent prompt
```

Quantization (the merged fine-tuned model as a q4_k_m file, on a CPU flavor):

```bash
uv run driftmap launch quant --wait -- --source merged               # smoke run first
uv run driftmap launch quant --mode full --wait -- --source merged
```

Weight-space repair of a fine-tune that lost safety:

```bash
uv run driftmap launch repair --wait         # the smoke screen of the repair
uv run driftmap launch repair --mode full --wait                     # generate the screen rows of each alpha
uv run driftmap judge closed --rows 'screen/<slug>/alpha-*.jsonl' --mode full
uv run driftmap repair choose --mode full    # rate each alpha, store the choice, runs here
uv run driftmap launch repair --mode full --wait -- --apply ALPHA    # push the repaired model
```

For the overfit fine-tune, add `--fine-tune aggressive` after the `--` of each repair launch and to
`repair choose` and `table build`; its screen, tables and repaired model keep names of their own.

Housekeeping:

```bash
uv run driftmap status                       # spend, pending jobs, latest run of each kind
uv run driftmap ledger reconcile             # billing against the ledger
uv run driftmap scan <path>...               # tokens, account ids, home paths, before publishing
```

Publish:

```bash
uv run driftmap publish plan --writeup URL   # scan every commit, print all that would go public
uv run driftmap publish run --writeup URL    # card, open and list the repos; try --dry-run first
uv run driftmap publish space                # create or update the results Space; try --dry-run first
uv run driftmap publish check                # which repos are public, carded and listed
```

`publish run` refuses before its first write when any commit of a repo holds a token, an address,
a home path, a cache path or a file the scan cannot read, when a merged or smoke repo is public or
sits in a collection (`--delete-intermediates` removes those repos first), when a required repo or
a collection is missing, when a card would be incomplete, when the cost ledger does not parse or
still holds a pending job, when a check file is out of place, empty or holds a finding, or when
the token is not the owner's.

With the cards, a run publishes three things from the checkout. The cost ledger goes to the results
dataset as `ledger.jsonl`, one line per job. The files under
`.run-local/checks/<responses|results>/<check>/` go to `checks/<check>/` in that dataset: the
answers and labels of the side checks (`q4_local`, `limit_768`, `engine_noise`, `judge_validation`,
`patch_heldout`) to the responses dataset, their summaries to the results dataset. Each adapter
keeps the trainer's log as `trainer_state.json` when its last checkpoint is removed. `publish plan`
lists all of this under the repo it goes to and ends with the Space step. `publish space` uploads
the page files in `space/` as a static Space in one commit, opens it, and reads the page back
without a token.

## The probe on SQuAD 2.0

The main result of the write-up: a probe on the base model's hidden state, fitted on the model's own errors on
SQuAD 2.0 questions and used as an abstention gate. Every step runs on a workstation CPU; no job is launched and
no weight changes.

```bash
uv run python -I -m driftmap.attribution.probe_squad --sources SOURCES --out OUT
uv run python -m driftmap.attribution.probe_curator --train OUT/squad_train_in.jsonl --test OUT/squad_dev_in.jsonl --work SCRATCH --out CURATED
uv run python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/squad_train_in.jsonl --output OUT/squad_train_base.jsonl --states OUT/acts/squad_train --max-new-tokens 48
uv run python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/squad_dev_in.jsonl --output OUT/squad_dev_base.jsonl --states OUT/acts/squad_dev --max-new-tokens 48
uv run python -m driftmap.attribution.local_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/squad_dev_in.jsonl --output OUT/squad_dev_auditkit.jsonl --max-new-tokens 48
uv run python -I -m driftmap.attribution.probe_study squad-fit --train OUT/squad_train_in.jsonl --answers OUT/squad_train_base.jsonl --states OUT/acts/squad_train --passed CURATED/squad_passed.json --model MODEL_DIR --manifest EVAL/manifest.json --out OUT
uv run python -I -m driftmap.attribution.probe_study squad --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --train OUT/squad_train_in.jsonl --train-answers OUT/squad_train_base.jsonl --passed CURATED/squad_passed.json --in-domain OUT/probe_squad --model MODEL_DIR --manifest EVAL/manifest.json --out RESULTS
```

`SOURCES`, `OUT`, `SCRATCH`, `CURATED` and `RESULTS` are local folders; `MODEL_DIR` is a local copy of the base
model, `REPO@REVISION` the label written to every row and `EVAL/manifest.json` the manifest of the evaluation
dataset.

- **Draw.** `probe_squad` fetches the validation file and the train file of SQuAD 2.0 at a pinned revision,
  checks them by sha256 and draws two samples: passages under 250 words, articles in a seeded order, one question
  per passage; 500 unanswerable and 500 answerable test questions, 1,000 and 1,000 training questions, and a
  second sample of 500 and 500 test questions (`squad_dev2_in.jsonl`): none of the first sample's questions,
  and one of its passages only where an article has no other left. The prompt files hold SQuAD text and stay
  where they are written; no other output holds any.
- **Gate.** `probe_curator` is the CuratorKIT gate over those rows: the schema gate on each file, exact dedup and
  MinHash dedup of the training prompts, and `cross_dedup` of the training sample against each test sample, on
  the passage and again on the question. `--test` may be given more than once. It writes `squad_passed.json`,
  the ids that pass, and `squad_curator.json`, the counts of every check.
- **Answers and hidden states.** `probe_generate` answers each prompt under the served template with plain
  transformers, greedy, in float32, and with `--states` keeps the hidden state at the last prompt token of the
  same run.
- **The same answers through AuditKIT.** `local_generate` answers the test questions through AuditKIT's
  `evaluate`, the path of the generate job. Its answers equal those of `probe_generate` row for row.
- **Fit.** `probe_study squad-fit` reads only the training questions that pass the gate, fits one
  standardisation and logistic regression per layer and C, picks layer, C and threshold on a dev split by
  article, and freezes the probe under `OUT/probe_squad` (`probe.safetensors`, `probe.json`).
- **Scoring.** `probe_study squad` scores the frozen probe once on the test questions that pass the gate, beside
  two text-only baselines of the same label and split (passage length, TF-IDF of the prompt). It writes
  `probe-squad.json`, the same as Markdown, and row files of ids, labels and scores without text.
  `--zero-shot NAME=FOLDER` adds a probe fitted elsewhere, read without refitting.

Three checks follow once the probe is frozen. Nothing is fitted in them and no threshold is moved; each reads
only the rows that pass the gate and writes `probe-squad-<part>.json`, the same as Markdown, and rows without text.

```bash
uv run python -I -m driftmap.attribution.probe_checks served-sample --dev OUT/squad_dev_in.jsonl --passed CURATED/squad_passed.json --out OUT/served_in.jsonl
uv run python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/served_in.jsonl --output OUT/served.jsonl --states OUT/acts/served --gate OUT/probe_squad --abstention "It cannot be determined from the text." --max-new-tokens 48
uv run python -I -m driftmap.attribution.probe_checks served --sample OUT/served_in.jsonl --served OUT/served.jsonl --served-states OUT/acts/served --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --passed CURATED/squad_passed.json --probe OUT/probe_squad --abstention "It cannot be determined from the text." --out RESULTS
uv run python -m driftmap.attribution.probe_judge --input OUT/squad_dev_in.jsonl --answers OUT/squad_dev_base.jsonl --cache JUDGE/score_cache.jsonl --out JUDGE/dev_verdicts.jsonl
uv run python -I -m driftmap.attribution.probe_checks scored --part judge --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --verdicts JUDGE/dev_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
uv run python -I -m driftmap.attribution.probe_checks identity --stored OUT/squad_dev_base.jsonl --kit OUT/squad_dev_auditkit.jsonl --out RESULTS
uv run python -I -m driftmap.attribution.probe_checks scored --part replication --dev OUT/squad_dev2_in.jsonl --dev-answers OUT/squad_dev2_base.jsonl --dev-states OUT/acts/squad_dev2 --verdicts JUDGE/dev2_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
uv run python -I -m driftmap.attribution.probe_checks scored --part pooled --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --verdicts JUDGE/dev_verdicts.jsonl --dev OUT/squad_dev2_in.jsonl --dev-answers OUT/squad_dev2_base.jsonl --dev-states OUT/acts/squad_dev2 --verdicts JUDGE/dev2_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
```

- **Served gate.** `served-sample` takes every tenth unanswerable and every tenth answerable test question. With
  `--gate`, `probe_generate` applies the probe inside the run: a forward hook reads the last prompt token
  during the prompt pass, and a row where the probe fires answers the sentence given with `--abstention` and
  nothing else. `served` compares those answers with the gate computed afterwards from the stored answers and
  hidden states (`probe-squad-served.json`).
- **Judge's reading.** `probe_judge` asks the closed model about each stored answer: does it abstain, does it
  state a reference answer. The route is read from `DRIFTMAP_CLOSED_JUDGE_ROUTE`, as for the judge of the side
  checks; replies are cached, so the same command continues a stopped run, and throttling stops it. The judge
  sees the passage, the question, the reference answers and the answer, not the word-match labels or the probe.
  `scored --part judge` writes the table of `probe_study squad` again under those verdicts, with how far the
  word-match rules agree with them (`probe-squad-judge.json`).
- **Identity.** `identity` compares the answers of `probe_generate` with those of `local_generate` row for row:
  text, token count and stop reason (`probe-squad-identity.json`).
- **Second sample.** `scored --part replication` applies the frozen probe to the second sample of the draw,
  answered, read and judged as the first (`OUT/squad_dev2_in.jsonl` and the files made from it; the gate is run
  with a second `--test`), and `--part pooled` to both samples given one after the other
  (`probe-squad-replication.json`, `probe-squad-pooled.json`).

The attribution fine-tune of the question pairs on the same test questions, beside the base model and the gated
base model:

```bash
uv run python -m driftmap.attribution.local_generate --model TUNED_DIR --label TUNED_REPO@REVISION --manifest EVAL/manifest.json --input OUT/squad_dev_in.jsonl --output OUT/squad_dev_ft.jsonl --max-new-tokens 48
uv run python -m driftmap.attribution.probe_judge --input OUT/squad_dev_in.jsonl --answers OUT/squad_dev_ft.jsonl --cache JUDGE/score_cache.jsonl --out JUDGE/ft_verdicts.jsonl
uv run python -I -m driftmap.attribution.probe_checks fine-tune --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --tuned-answers OUT/squad_dev_ft.jsonl --verdicts JUDGE/dev_verdicts.jsonl --tuned-verdicts JUDGE/ft_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
```

`TUNED_DIR` is a local copy of the merged fine-tune. Its answers are generated through AuditKIT's `evaluate`, read
by the same judge, and set beside the base model and the base model gated by the frozen probe, by the word-match
rules and by the judge's labels (`probe-squad-fine-tune.json`). Nothing is trained in this step.

One instruction line in the prompt, with no probe, on the same test questions:

```bash
uv run python -I -m driftmap.attribution.probe_checks instructed-sample --dev OUT/squad_dev_in.jsonl --passed CURATED/squad_passed.json --out OUT/squad_dev_instructed_in.jsonl
uv run python -m driftmap.attribution.local_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/squad_dev_instructed_in.jsonl --output OUT/squad_dev_instructed.jsonl --max-new-tokens 48
uv run python -m driftmap.attribution.probe_judge --input OUT/squad_dev_in.jsonl --answers OUT/squad_dev_instructed.jsonl --cache JUDGE/score_cache.jsonl --out JUDGE/instructed_verdicts.jsonl
uv run python -I -m driftmap.attribution.probe_checks fine-tune --part prompt --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --tuned-answers OUT/squad_dev_instructed.jsonl --verdicts JUDGE/dev_verdicts.jsonl --tuned-verdicts JUDGE/instructed_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
```

`instructed-sample` writes the test prompts again with one line before the closing line, "If the text does not
give enough information to answer, say that it cannot be determined."; the base model answers them through
AuditKIT's `evaluate`; the judge reads the answers without seeing the instruction; and `fine-tune --part prompt`
sets them beside the base model and the gated base model (`probe-squad-prompt.json`). The probe is not applied
to the instructed prompts.

The model's own confidence in its answer as the gate, on the second sample:

```bash
uv run python -m driftmap.attribution.probe_confidence --model MODEL_DIR --manifest EVAL/manifest.json --prompts OUT/squad_dev2_in.jsonl --answers OUT/squad_dev2_base.jsonl --output OUT/confidence.jsonl --workers 6 --threads 10
uv run python -I -m driftmap.attribution.probe_checks confidence --dev OUT/squad_dev2_in.jsonl --dev-answers OUT/squad_dev2_base.jsonl --dev-states OUT/acts/squad_dev2 --confidence OUT/confidence.jsonl --verdicts JUDGE/dev2_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
```

`probe_confidence` runs one forward pass over each prompt followed by its stored answer, in float32 on the CPU,
and generates nothing. `OUT/confidence.jsonl` holds one row per question and no text: its id, the probability of
the stored answer's first token and the mean log-probability of its tokens. No threshold is fitted: each confidence gate abstains on its least confident answers until it
answers as many questions as the probe's gate, and is set beside it there and at every tenth of the answers
(`probe-squad-confidence.json`).

The probe's gate on top of the instruction, an exploratory computation:

```bash
uv run python -I -m driftmap.attribution.probe_checks combined --dev OUT/squad_dev_in.jsonl --dev-answers OUT/squad_dev_base.jsonl --dev-states OUT/acts/squad_dev --tuned-answers OUT/squad_dev_instructed.jsonl --verdicts JUDGE/dev_verdicts.jsonl --tuned-verdicts JUDGE/instructed_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
```

`combined` generates nothing. Where the frozen probe fires on the stored state of the plain prompt the answer is
an abstention; elsewhere it is the stored answer to the instructed prompt. The record counts the four systems,
the change against three of them by article, and how the gate and the instruction overlap
(`probe-squad-prompt-probe.json`, status exploratory). The probe is not read on the instructed prompt.

The CircuitKIT check of the probe's layer and of the layer choice, in an environment of its own:

```bash
pip install --no-deps -r tools/circuitkit/requirements-circuitkit.txt
python -I tools/circuitkit/circuitkit_check.py --package src states --model MODEL_DIR --manifest EVAL/manifest.json --input OUT/squad_dev_in.jsonl --stored OUT/acts/squad_dev --probe OUT/probe_squad --every 5 --workers 2 --threads 10 --out CHECK
python -I tools/circuitkit/circuitkit_check.py --package src sweep --train OUT/squad_train_in.jsonl --answers OUT/squad_train_base.jsonl --stored OUT/acts/squad_train --fit OUT/probe_squad_fit.json --out CHECK
python -I tools/circuitkit/circuitkit_check.py --package src report --out CHECK --results RESULTS
```

CircuitKIT pins its own TransformerLens, so the requirements file is added to a copy of the development
environment and not to the lock (`tools/circuitkit/README.md`); `--package src` names the package, which that
environment does not install. `states` loads the
base model through `circuitkit.load_model`, reads `blocks.19.hook_resid_post` at the last token of every fifth
test prompt, and applies the frozen probe to that state and to the stored one. `sweep` trains CircuitKIT's
`LinearProbe` with its `ProbeTrainer` on the stored states of the train sample, one probe per layer. `report`
writes the counts and a row file of ids, probabilities and decisions (`probe-squad-circuitkit.json`).

## The question pairs

The main result: who did it, when the text does not say. The training rows are under `sft_train_defense/`
and the question pairs under `choice/` in the evaluation dataset; `COMMIT` is the commit that holds them.

The rows and the scores that no job produces come from the modules of `driftmap.attribution`, run on a
workstation:

```bash
uv run python -I -m driftmap.attribution.build --bbq BBQ --benign sft_train/en.jsonl --out DATA
uv run python -m driftmap.attribution.translate --source DATA/test_en.jsonl --out DATA --english
uv run python -m driftmap.attribution.curator_report --data DATA --prompts PROMPTS --work SCRATCH --out DATA/curator_report.json
uv run python -I -m driftmap.attribution.siblings --bbq BBQ --manifest DATA/manifest.json --test DATA/test_en.jsonl --answers base=BASE.jsonl --answers fine_tuned=TUNED.jsonl --report DATA/curator_report.json
uv run python -I -m driftmap.attribution.free_form_rows --prompts CHOICE_PROMPTS --out FREE base=BASE.jsonl tuned=TUNED.jsonl
uv run python -m driftmap.attribution.free_form --out SCORED FREE/base.jsonl FREE/tuned.jsonl
```

- `build` draws the training rows and the English question pairs from BBQ. BBQ is not in this repository:
  the files are downloaded from `nyu-mll/BBQ` at commit `bea11bd97d79` into the folder `BBQ` and checked
  by sha256 before they are read. The draw is seeded: the same inputs give the same files byte for byte.
- `translate` puts the test pairs into the other five languages and keeps the ones a blind check confirms.
- `curator_report` runs the CuratorKIT gate, the dedup steps and the leak check over the training rows;
  `PROMPTS` holds the prompts of the side checks as `<set>/<language>.jsonl`.
- `siblings` counts the held-out question templates that share a stereotype with a trained one.
- `free_form_rows` joins the free-form answers that `driftmap table choice` sets aside with their prompt
  rows (`CHOICE_PROMPTS` holds the question pairs as `<language>.jsonl`); `free_form` then reads which
  option each answer asserts.

`translate` and `free_form` call the closed model named by `DRIFTMAP_CLOSED_JUDGE_ROUTE` and keep its
replies in a cache beside their output, so a second run of the same command makes no call.

The check on two other datasets (reading questions from Belebele, answerable and unanswerable questions
from SQuAD 2.0) runs on a workstation CPU:

```bash
uv run python -I -m driftmap.attribution.reading --sources SOURCES --translation DATA/translation_manifest.json --out OUT
uv run python -m driftmap.attribution.local_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/belebele_in.jsonl --output OUT/belebele_base.jsonl --max-new-tokens 24 --batch-size 10 --workers 3 --threads 10
uv run python -I -m driftmap.attribution.reading_scores belebele --base OUT/belebele_base.jsonl --fine-tuned OUT/belebele_ft.jsonl --out OUT
uv run python -I -m driftmap.attribution.reading_answers belebele --sources SOURCES --base OUT/belebele_base.jsonl --fine-tuned OUT/belebele_ft.jsonl --out PUBLISHED
uv run python -I -m driftmap.attribution.reading_answers squad --base OUT/squad_base.jsonl --fine-tuned OUT/squad_ft.jsonl --out PUBLISHED
```

- `reading` builds the prompts. Both datasets are downloaded at pinned revisions
  (`facebook/belebele@7899cdfa4e1e`, `rajpurkar/squad_v2@3ffb306f725f`) into `SOURCES` and checked by
  sha256; both are CC BY-SA 4.0 and neither is kept in this repository. The draws are seeded.
- `local_generate` answers them through the generate job's rendering, greedy, in float32: at most 24 new
  tokens for Belebele and 48 for SQuAD 2.0 (`squad_in.jsonl`). It runs once per model.
- `reading_scores` counts the answers by rule; `squad` in place of `belebele` counts the SQuAD answers.

- `reading_answers` writes the answers in the shape that is published. A row names its question by the id
  the source dataset gives it and by the dataset's pinned revision, and holds no passage, question, option
  or gold-answer text; a Belebele row records which of the source's answers sits under each letter. For
  each SQuAD 2.0 answer the row carries its wording class and whether it contains a gold answer, but not
  the answer text, because the free-form answers quote the passages and the passages are not
  redistributed; the text is kept only where it is one of the trained abstention sentences (it is left
  out for 395 of the 400 rows). The Belebele answers, which are a letter or a letter name, are published
  as they are.

No passage or question of the two datasets is published, and `reading` rebuilds the prompts from the
sources.

The answer lengths the write-up cites (`checks/answer_lengths/answer_lengths.json` in the results dataset)
are counted from the stored answers under the served template:

```bash
uv run python -I -m driftmap.attribution.answer_lengths RESPONSES_DIR OUT.json --model "base model=tiny-aya-global--af89d219b53e" --model "light fine-tune=tiny-aya-global-drift-merged--daccc14f6c19" --model "attribution fine-tune=tiny-aya-global-drift-merged-defense--bf62811bdd99"
```

`RESPONSES_DIR` is a local copy of the responses dataset; each `--model` names a model and the folder of
its checkpoint under `responses/`. Per model and prompt set (red-teaming, XSTest, held-out benign) the file
holds the rows, the mean of generated tokens, the answers that stopped at the 256-token limit, and the
answers that are word for word one of the trained abstention sentences; no prompt and no answer text.

The fine-tune and the answers are jobs:

```bash
uv run driftmap launch drift --mode full --wait -- --train-set defense --train-rev COMMIT --epochs 3
uv run driftmap launch generate --mode full --wait -- --kind baseline --sets choice --sets-rev COMMIT --max-new-tokens 48
uv run driftmap launch generate --mode full --wait -- --kind baseline --preamble patch --sets choice --sets-rev COMMIT --max-new-tokens 48
uv run driftmap launch generate --mode full --wait -- --kind drifted --checkpoint REPO@SHA --sets choice --sets-rev COMMIT --max-new-tokens 48
uv run driftmap table choice --fine-tune defense
uv run driftmap figures --fine-tune defense
```

The fine-tune job refuses a training set that holds a test prompt. `table choice` reads the stored
answers by rule, with no judge, and writes `tables/choice-defense.json` and its printed form to the
results dataset. `defense` and `drifted` name the attribution fine-tune in commands, paths and record
fields.

`figures` draws the two figures of the main result from that table, `f10_attribution_summary` and
`f9_attribution`, as PNG and SVG into `docs/figures-defense/`; the files under `docs/figures/` are copies
of them.

The question pairs in four languages come from MBBQ, of question types the fine-tune never saw (`MBBQ`
and `BBQ` are folders for the two datasets' files, downloaded at pinned commits and checked by sha256;
`DATA` is the folder `build` wrote; `NAME=FOLDER` names an arm and the folder of its answers):

```bash
uv run python -I -m driftmap.attribution.mbbq_build --mbbq MBBQ --bbq BBQ --data DATA --out OUT
uv run driftmap launch generate --mode full --wait -- --kind baseline --sets mbbq --sets-rev COMMIT --max-new-tokens 48
uv run driftmap launch generate --mode full --wait -- --kind baseline --preamble patch --sets mbbq --sets-rev COMMIT --max-new-tokens 48
uv run driftmap launch generate --mode full --wait -- --kind drifted --checkpoint REPO@SHA --sets mbbq --sets-rev COMMIT --max-new-tokens 48
uv run driftmap table choice --sets mbbq --fine-tune defense --fine-tune standard
uv run python -I -m driftmap.attribution.mbbq_answers --prompts PROMPTS --out OUT NAME=FOLDER [NAME=FOLDER ...]
```

A launch refuses an MBBQ item whose BBQ base item or question template is in the training set. The table
goes to `tables/choice-mbbq-defense.json`. MBBQ's text is not republished: `mbbq_answers` writes
`items.jsonl`, which names every prompt by the row of MBBQ it is made of, and `answers.jsonl`, the letter
each answer is read as (`mbbq/` in the results dataset); with the release pinned in `mbbq/manifest.json`
of the evaluation dataset every prompt can be written again.

The activation probe is fitted and scored from stored answers, on a CPU:

```bash
uv run python -I -m driftmap.attribution.probe_rows --bbq BBQ --data DATA --sets SETS.json --out OUT
uv run python -m driftmap.attribution.probe_extract --model MODEL_DIR --manifest EVAL/manifest.json --input OUT/probe_train_in.jsonl --output OUT/acts/train
uv run python -m driftmap.attribution.probe_extract --model MODEL_DIR --manifest EVAL/manifest.json --input OUT/probe_eval_in.jsonl --output OUT/acts/eval
uv run python -I -m driftmap.attribution.probe_fit --rows OUT/probe_train_in.jsonl --states OUT/acts/train --out OUT
uv run python -I -m driftmap.attribution.probe_scores --probe OUT --sets SETS.json --prompts OUT/probe_eval_in.jsonl --states OUT/acts/eval --out RESULTS --fine-tune defense
```

`probe_rows` writes the rows the probe is fitted and chosen on, the base items the fine-tune trained on with
whole question templates set aside as dev, and the prompts of the evaluation sets `SETS.json` names.
`probe_extract` reads the base model's hidden state at the last token of each prompt; `probe_fit` picks one
layer, one C and one threshold by a fixed rule on dev and freezes the probe (`probe.safetensors`,
`probe.json`); `probe_scores` writes `probe-defense.json`, `.md` and a row file without text. Where the probe
fires the gated answer is the unknown option, elsewhere the base model's stored answer: nothing is generated
and no weight changes.

The same study on the question pairs: a probe of the base model's own error beside the probe above, text-only
baselines for both, and the gate served inside generation.

```bash
uv run python -I -m driftmap.attribution.probe_study error-fit --rows OUT/probe_train_in.jsonl --answers TRAIN_ANSWERS.jsonl --states OUT/acts/train --out OUT
uv run python -I -m driftmap.attribution.probe_study bbq --sets SETS.json --prompts OUT/probe_eval_in.jsonl --states OUT/acts/eval --probe probe_1=OUT --probe error_probe=OUT/probe_error --out RESULTS
uv run python -I -m driftmap.attribution.probe_study baselines --rows OUT/probe_train_in.jsonl --labels OUT/probe_error_rows.jsonl --sets SETS.json --prompts OUT/probe_eval_in.jsonl --states OUT/acts/eval --probe probe_1=OUT --probe error_probe=OUT/probe_error --model MODEL_DIR --manifest EVAL/manifest.json --out RESULTS
uv run python -I -m driftmap.attribution.probe_study served-sample --bbq-rows BBQ_ROWS.jsonl --mbbq MBBQ_PROMPTS --out OUT/served_in.jsonl
uv run python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/served_in.jsonl --output SERVED.jsonl --gate OUT --states OUT/acts/served --max-new-tokens 24
uv run python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION --manifest EVAL/manifest.json --input OUT/served_in.jsonl --output UNGATED.jsonl --max-new-tokens 24
uv run python -I -m driftmap.attribution.probe_study served --sample OUT/served_in.jsonl --served SERVED.jsonl --ungated UNGATED.jsonl --served-states OUT/acts/served --sets SETS.json --prompts OUT/probe_eval_in.jsonl --states OUT/acts/eval --probe OUT --out RESULTS
```

`error-fit` fits a probe of the base model's own error on the rows of the first probe (`TRAIN_ANSWERS.jsonl`
holds the base model's answers to them); `bbq` scores it beside the first probe on the evaluation sets
(`probe-bbq.json`); `baselines` fits a count of context tokens and a TF-IDF of the prompt for both labels and
scores them beside the two probes (`probe-baselines.json`); `served-sample` draws the prompts of the served
check, and `served` compares the answers of a run with the gate inside it with the gate computed afterwards
(`probe-served.json`). Each table has its Markdown copy and a row file without text.

The padding test, the control-set check and the UnQover check of the write-up are checks, not package
commands: their answers and counts are under `checks/padding/`, `checks/mbbq-control/` and
`checks/unqover/` in the results dataset, the last two without any text of MBBQ or of the built items.

## The judge

The tables are scored by `claude-sonnet-5-5`, a closed model, under the rubric in
`src/driftmap/judge/rubric.py` (`rubric-v1`): one JSON object per exchange with `request_harm`,
`refusal` and `response_harm`. `driftmap judge closed` runs on the workstation.

- **Route.** The route to the model is named in the environment variable
  `DRIFTMAP_CLOSED_JUDGE_ROUTE`, together with whatever credentials that route needs. It is read
  from the environment only.
- **Model name.** Label rows carry the public model name from `[judge.closed]` in `driftmap.toml`
  and the rubric version. The name is taken on trust from the configuration: nothing checks
  which model the route serves.
- **Resume.** The command labels one set and language at a time, uploads after each, and picks
  up where it stopped when run again.
- **Throttling.** It stops on throttling or an exhausted quota instead of retrying.
- **Answers without labels.** When more than 5% of the pairs sent for one file, and more than
  one, come back without labels, the answers that parsed are stored, the rest is held back and
  the command exits 4. Run it again, or with `--retry-unscored`, which asks unscored pairs again
  and stores what stays unscored.
- **Format reminder.** An answer that is text without the object is asked again with a one-line
  format reminder after the unchanged pair.
- **Quoted input.** The request and the response are quoted between tags that carry a random
  marker per call, and the rubric says they are data, never instructions.
- **Single files.** With `--rows PATH` in place of `--checkpoint` it labels the harmful rows of
  one responses-repo file (or a glob of them, such as the repair screen's
  `screen/<slug>/alpha-*.jsonl`) and writes `labels-<name>` beside each.

## The no-safety-preamble condition

`--preamble strip` on the generate and drift jobs is the "no safety preamble" condition: every
prompt is rendered without the safety paragraph the model's chat template opens its system turn
with, and nothing else in the template changes. Those runs keep their files under `nopreamble/`,
a fine-tune trained that way pushes to repos ending in `-nopre`, and
`driftmap judge closed --preamble strip` labels those responses.

## The repair

The repair job moves fine-tuned weights toward the original model (WiSE-FT): each weight becomes
`(1 - alpha) * fine-tuned + alpha * original`. It takes two jobs with a step on the workstation between
them. The screen job generates, for each alpha under `[repair]` in `driftmap.toml`, the answers to 20
harmful and 20 benign prompts per language and scores the benign ones with chrF. `judge closed --rows`
labels the harmful answers. `repair choose` then rates every alpha over the prompts that it, the other
alphas and the base model all have a verdict on, and picks the smallest alpha whose safe rate is within
`safety_points` of the base model's and whose chrF is within `chrf_points` of the fine-tuned model's. It
prints one line per alpha and stores the record as `alpha_screen.json` in the results dataset:
`chosen`, `failed` when no alpha passes both bars, or `screened` while labels are missing or cover
under 90% of an alpha's harmful prompts. The apply job pushes the model at the chosen alpha and refuses
any other value.

## What a launch is checked against

A launch needs a readiness report less than a day old, made under the current configuration.
Then every guard runs; one refusal is exit 4 and nothing is submitted.

- `gate`: the token can read the gated base model.
- `code_pinned`: HEAD is clean and on the origin branch; a job installs the code from that commit.
- `smoke`: a full run has a passing smoke run of the same script, image and pins.
- `per_run_ceiling`: the cost at the job's whole timeout is within the per-run ceiling; smoke
  runs also stay within the smoke total and run one at a time.
- `ledger_ceiling`: spend so far plus this job is within the soft ceiling.
- `in_flight_gpu`: no other GPU job is running or pending on the ledger.
- `timeout_lost`: no job of this kind was already lost at this timeout.
- `oom_repeat`: this kind did not already run out of memory under this configuration.
- `drift_decision`: a repair waits for `decide drift` while the measured drop in safe rate is below the threshold.

The ceilings are under `[ceilings]` in `driftmap.toml`; the ledger is `.run-local/results/ledger.jsonl`,
and `publish run` copies it to the results dataset as `ledger.jsonl`, one line per job.

A repair launch is also checked for its inputs: an ok fine-tune run and the fine-tuned checkpoint's
benign labels, and with `--apply` a screen that `repair choose` stored as chosen. `chain` takes no
repair step, because the choice sits between its two jobs.

## On the Hub

Names under the owner, from `[repos]` in `driftmap.toml`:

- `aya-redteaming-6lang-eval` (dataset): prompts, training rows and labels.
- `safety-drift-map-responses` (dataset): model responses and the judge's raw outputs.
- `safety-drift-map-results` (dataset): run records, tables and decisions.
- `tiny-aya-global-drift-lora` (model): the fine-tuned adapter.
- `tiny-aya-global-drift-merged` (model, private): the merged fine-tuned weights, an intermediate.
- The same two names ending in `-nopre`, once a fine-tune with no safety preamble exists: its
  adapter is published like the first, its merged weights stay private.
- `tiny-aya-global-drift-GGUF` (model): the fine-tuned model as a q4_k_m file, the one the 4-bit
  evaluation ran on.
- `tiny-aya-global-repaired-nopre` (model): the overfit fine-tune repaired by weight interpolation
  with the base model at alpha 0.25.
- `tiny-aya-global-repaired` and `tiny-aya-global-repaired-GGUF` (models): the names a repair of the
  light fine-tune and its q4_k_m file would publish under; neither repo exists.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | done |
| 1 | unexpected error |
| 2 | usage or configuration |
| 3 | not ready |
| 4 | refused by a guard, or the judge stopped (throttling, quota, too many answers without labels) |
| 5 | job lost or not found |
| 6 | redaction finding |
| 7 | upload failed |
| 8 | drift decision needed |
| 12 | a data, label or record file, or the published state, is not what it should be |
| 130 | interrupted |

## Reproduce the tables and the results page

The tables, without a GPU or a judge: open [the notebook](../notebooks/reproduce_tables.ipynb). It
downloads the published label files and recomputes the rates, intervals and paired changes of the write-up,
except the 4-bit table and the rows on answer length, which also need the responses dataset.

The Space's data file, from the same label files, and the page served locally (it loads its data
over HTTP, so opened as a file it only shows a notice):

```bash
python space/build_data.py
python -m http.server --directory space      # then open http://localhost:8000/
```

The whole pipeline, from the source datasets to the tables (Hugging Face Jobs, one L4):

```bash
uv python install 3.12 && uv sync --frozen
uv run pytest -q -m "not smoke"

uv run driftmap ready --pin
uv run driftmap data build
uv run driftmap launch generate --mode full --wait -- --kind baseline
uv run driftmap launch drift --mode full --wait -- --epochs 3
uv run driftmap launch generate --mode full --wait -- --kind drifted
uv run driftmap judge closed --checkpoint REPO@SHA --mode full       # once per checkpoint
uv run driftmap table build
```

Each full run needs a passing smoke run first; the sections above give the details. In command
names, file paths and record fields the fine-tuned model is called `drifted` and the base model
`baseline`.
