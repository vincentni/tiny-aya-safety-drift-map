# Tiny Aya Global answers 97% of questions its passage cannot answer; a probe on its activations cuts wrong answers from 56% to 29%, 8 points better than one instruction line (36%), while 80% of answerable questions are still answered.

[![tests](https://github.com/vincentni/tiny-aya-safety-drift-map/actions/workflows/tests.yml/badge.svg)](https://github.com/vincentni/tiny-aya-safety-drift-map/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vincentni/tiny-aya-safety-drift-map/blob/main/notebooks/reproduce_tables.ipynb)

Cohere's Tiny Aya Global (3.35B) is asked questions about passages, half of which the passage cannot answer
(SQuAD 2.0). A probe on its layer-20 activations, trained on the model's own errors, flags the answers it is about
to get wrong; the model abstains where the probe fires and answers as before elsewhere. Lexsi AI Safety &
Alignment Hackathon 2026, Track 2b: Hallucination detection and abstention.

| On 1,000 SQuAD 2.0 dev questions, 500 of them unanswerable | Wrong answers | Wrong answers as read by a closed judge | Unanswerable questions answered, of 500 | Answerable questions still answered, of 500 |
|---|---|---|---|---|
| Base model | 560 (56.0%) | 505 (50.5%) | 485 | 497 |
| Abstaining on every question | 500 (50.0%) | 500 (50.0%) | 0 | 0 |
| One instruction line in the prompt, no probe | 364 (36.4%) | 335 (33.5%) | 201 | 378 |
| **Probe on the activations as the gate** | 286 (28.6%) | 248 (24.8%) | 140 | 401 |
| Probe gate, else the instructed answer (exploratory) | 276 (27.6%) | 256 (25.6%) | 68 | 320 |

*Wrong: an unanswerable question answered, or an answerable one answered incorrectly or abstained on. Word match is the rule set in advance; the judge's reading, the instruction line and the last row were added after the main result, and the last row is exploratory. Probe against base model: -27.4 points (95% CI over articles -32.3 to -22.8); probe against the instruction line: 7.8 points (95% CI over articles 3.89 to 11.9, p = 0.008) better.*

- **560 → 286 wrong answers** of 1,000 official SQuAD 2.0 dev questions with the probe gate (-27.4 points, 95% CI over articles -32.3 to -22.8; 35 of 35 articles better). No weight changes.
- **One instruction line gets about two thirds of that for free**: 560 → 364 with no probe. The probe is 7.8 points (95% CI over articles 3.89 to 11.9, p = 0.008) better and keeps more answerable questions (401 against 378); combining the two (exploratory) does not lower wrong answers further.
- **Replicated on a second sample of questions**, mostly on the same passages (838 of 1,000 shared): 575 → 315; both samples together 1,135 → 601 of 2,000.
- **The failure**: the base model answers 485 of 500 questions its passage cannot answer.
- **It still answers**: 401 of 500 answerable questions get an answer, 354 with a gold answer (base model 497 and 425).
- **Not a text cue, and more than output confidence**: AUC against the model's error 0.8246 for the probe, 0.4978 for passage length, 0.5545 for TF-IDF; on the second sample the model's own output confidence 0.6951 at best, the probe 0.7853.
- **Three hackathon kits in the result**: CuratorKIT gates the questions, AuditKIT generates the scored answers, CircuitKIT reads the same hidden states (same probe decision on 200 of 200 prompts) and finds the same layer.
- **Served**: a forward hook that applies the probe during generation gives the offline gate's answers on 100 of 100 dev prompts.
- **Where it pays**: abstaining on everything gives 500 wrong of 1,000; at its rates the gate cuts wrong answers only where more than 17.1% of the questions are unanswerable (20.2% by the judge's reading).
- **Data**: why each dataset was used and how it was built is a table in the write-up's [data section](docs/article.md#sources-of-data); the constructed sets of the secondary evidence are in the [question-pair page](docs/question-pairs.md).
- **More in the write-up**: what does not carry from one kind of unanswerable question to another, and the secondary evidence on question pairs.
- **The limits**: English only; one probe, one seed, one prompt wording; word-match scoring (a closed judge's labels give 505 → 248); the errors it flags are mostly answered unanswerable questions.

[Write-up](docs/article.md) | [Results Space](https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map) | [Models](https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-models) | [Datasets](https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-data) | [Notebook](notebooks/reproduce_tables.ipynb) | [Commands](docs/commands.md)

**Content warning:** the side checks use red-teaming prompts, and the datasets hold harmful requests and
answers. Nothing harmful is quoted in this repository; the results page withholds the text of answers judged harmful.

**For judges:** the probe runs on a CPU and changes no weight. The [results Space](https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map) shows the tables without running a model; the [notebook](notebooks/reproduce_tables.ipynb) recounts the main table from one row per question, without a GPU.

![A walk through the results page: the headline, the main table, the judge's reading and a second sample, the probe beside an instruction line, a fine-tune and output confidence, risk against coverage, probes fitted on another kind of question, and where to click](docs/figures/explorer.gif)

*The [results Space](https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map) in 26 seconds: the headline, the main table, the judge's reading and a second sample of questions, the probe beside an instruction line, a fine-tune and output confidence, risk against coverage, probes fitted on another kind of question, and where to click.*

## Results

Served chat template, SQuAD 2.0 dev questions. Method, limits and costs: the [write-up](docs/article.md). The probe's fit, the baselines and the statistics are scikit-learn and scipy.

| Score, trained on the same questions | AUC against the model's error |
| --- | --- |
| Probe on layer 20 | 0.8246 |
| Passage length | 0.4978 |
| TF-IDF of the prompt | 0.5545 |

## Quick start

```bash
uv python install 3.12 && uv sync --frozen
uv run pytest -q -m "not smoke"                  # the offline tests
python space/build_data.py                       # the results page's data, from the public labels
python -m http.server --directory space          # the results page at http://localhost:8000/
uv run driftmap --help                           # the pipeline behind the numbers
```

Every stored answer can be recounted without a GPU in [the notebook](notebooks/reproduce_tables.ipynb);
every pipeline command is in the [commands reference](docs/commands.md).

## Repo layout

| Path | What it holds |
| --- | --- |
| `src/driftmap/` | the package: data build, the question pairs (`attribution`), generation, judge, tables and figures, repair, quantization, publishing, the job guards and the cost ledger |
| `jobs/`, `tests/` | the entry points a Hugging Face Job runs; offline unit tests |
| `tools/circuitkit/` | the CircuitKIT check of the probe's hidden states |
| `notebooks/reproduce_tables.ipynb` | recounts the stored answers and recomputes the side-check tables |
| `space/` | the static results page and the script that builds its data file |
| `docs/` | the write-up, the abstract, the detail page on harmful requests, the figures and the commands reference |

## Licences and use

- Code: Apache-2.0, see `LICENSE`.
- Tiny Aya Global is cc-by-nc-4.0. Every weight derived from it here is under the same licence:
  non-commercial use only.
- Source datasets: BBQ and XSTest are cc-by-4.0, `aya_redteaming` and `aya_dataset` are Apache-2.0; the
  evaluation set built from them keeps those terms. Belebele and SQuAD 2.0 are cc-by-sa-4.0: no passage,
  question or gold answer of theirs is redistributed; the published rows are keyed by the datasets' own question ids.
- The adapters are research artifacts, published so the measurements can be repeated. They are
  evaluated only as described here and are not certified for deployment.
