# driftmap

Measures how benign fine-tuning erodes refusal behaviour in Cohere's Tiny Aya Global,
language by language, repairs it with SafeTune, and re-checks the repair after 4-bit
quantization. Written for the Lexsi hackathon 2026. The write-up is `docs/article.md`.

The pipeline in one line: build the evaluation set, generate and judge the baseline,
fine-tune (the drift), generate and judge the drifted model, repair, quantize, build the tables.

Jobs run on Hugging Face Jobs. Every dataset, model and result lives on the Hub under the
owner named in `driftmap.toml`; local state is in `.run-local/`, which is not tracked.

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

    uv python install 3.12 && uv sync --frozen
    uv run pytest -q -m "not smoke"

The tests are offline. Commands that reach the Hub read `HF_TOKEN`, or the token stored by
`hf auth login`, and expect it to belong to the owner.

## Commands, in the order they are run

    uv run driftmap ready --pin                  # pin the source revisions, run the readiness checks
    uv run driftmap data build --dry-run         # build the evaluation set locally, print the counts
    uv run driftmap data build                   # upload it and pin the commit
    uv run driftmap plan generate                # the job, its cost at timeout, every guard verdict
    uv run driftmap launch generate --wait -- --kind baseline            # smoke run first
    uv run driftmap launch judge --wait --after <generate run id>
    uv run driftmap launch generate --mode full --wait -- --kind baseline
    uv run driftmap launch judge --mode full --wait --after <generate run id>
    uv run driftmap launch drift --wait          # the smoke drift
    uv run driftmap launch drift --mode full
    uv run driftmap wait <job id>                # follow a job, record what it cost
    uv run driftmap chain generate judge --target drifted                # two steps, one after the other
    uv run driftmap status                       # spend, pending jobs, latest run of each kind
    uv run driftmap ledger reconcile             # billing against the ledger
    uv run driftmap table build                  # tables from the labels: saved, printed, uploaded
    uv run driftmap decide drift --proceed --note "why"                  # when table build exits 8
    uv run driftmap judge reparse --checkpoint REPO@SHA                  # after a parser change, no GPU
    uv run driftmap scan <path>...               # tokens, account ids, home paths, before publishing

`--mode` is `smoke` by default for `plan` and `launch`: a short rehearsal on a few rows.
Arguments after `--` go to the job script. Every command has `--help`.

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
- `drift_decision`: repair and steer wait for `decide drift` while the measured drift is below the threshold.

The ceilings are under `[ceilings]` in `driftmap.toml`; the ledger is `.run-local/results/ledger.jsonl`.

## On the Hub

Names under the owner, from `[repos]` in `driftmap.toml`:

- `aya-redteaming-6lang-eval` (dataset): prompts, training rows and labels.
- `safety-drift-map-responses` (dataset): model responses and the judge's raw outputs.
- `safety-drift-map-results` (dataset): run records, tables and decisions.
- `tiny-aya-global-drift-lora` (model): the drifted adapter.
- `tiny-aya-global-drift-merged` (model, private): the merged drifted weights, an intermediate.
- `tiny-aya-global-repaired` and `tiny-aya-global-repaired-GGUF` (models): the repaired model and its q4_k_m file.

## Licences and use

- Code: Apache-2.0, see `LICENSE`.
- Tiny Aya Global is cc-by-nc-4.0. Every weight derived from it here (adapter, merged, repaired,
  GGUF) is under the same licence: non-commercial use only.
- Source datasets: `aya_redteaming` and `aya_dataset` are Apache-2.0, XSTest is cc-by-4.0; the
  evaluation set built from them keeps those terms.
- The drifted adapter is a research artifact only. It is a model made less safe on purpose and
  is published so the measurement can be repeated. Do not deploy it or serve it to users.

## Exit codes

0 done, 1 unexpected error, 2 usage or configuration, 3 not ready, 4 refused by a guard,
5 job lost or not found, 6 redaction finding, 7 upload failed, 8 drift decision needed,
12 a data, label or record file is not what it should be, 130 interrupted.
