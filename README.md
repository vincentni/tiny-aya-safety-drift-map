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

    driftmap ready --pin                  # pin the source revisions, run the readiness checks
    driftmap data build --dry-run         # build the evaluation set locally, print the counts
    driftmap data build                   # upload it and pin the commit
    driftmap plan generate                # the job, its cost at timeout, every guard verdict
    driftmap launch generate --wait -- --kind baseline            # smoke run first
    driftmap launch judge --wait --after <generate run id>
    driftmap launch generate --mode full --wait -- --kind baseline
    driftmap launch judge --mode full --wait --after <generate run id>
    driftmap launch drift --mode full
    driftmap wait <job id>                # follow a job, record what it cost
    driftmap chain generate judge --target drifted                # two steps, one after the other
    driftmap status                       # spend, pending jobs, latest run of each kind
    driftmap ledger reconcile             # billing against the ledger
    driftmap table build                  # tables from the labels: printed, saved, uploaded
    driftmap decide drift --proceed --note "why"                  # when table build exits 8
    driftmap judge reparse --checkpoint REPO@SHA                  # after a parser change, no GPU
    driftmap scan <path>...               # tokens, account ids, home paths, before publishing

`--mode` is `smoke` by default for `plan` and `launch`: a short rehearsal on a few rows. A
`full` run is refused until the same job has passed its smoke run. Arguments after `--` go
to the job script. Every command has `--help`.

## Spending rules

- A launch needs a readiness report less than a day old, made under the current configuration.
- A job is held on the ledger at the cost of its whole timeout until it ends; the ceilings
  under `[ceilings]` in `driftmap.toml` are checked against that before every submit.
- One GPU job runs at a time. The ledger is `.run-local/results/ledger.jsonl`.

## Exit codes

0 done, 1 unexpected error, 2 usage or configuration, 3 not ready, 4 refused by a guard,
5 job lost or not found, 6 redaction finding, 7 upload failed, 8 drift decision needed.

Licensed under Apache-2.0, see `LICENSE`.
