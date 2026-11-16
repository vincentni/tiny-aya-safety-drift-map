# Tiny Aya Global answers 97% of questions its passage cannot answer. A probe on its activations cuts wrong answers from 56% to 29% with no weight changes, 8 points better than one instruction line (36%); a LoRA fine-tune on the same 2,000 questions reaches 17%.

[![tests](https://github.com/vincentni/tiny-aya-safety-drift-map/actions/workflows/tests.yml/badge.svg)](https://github.com/vincentni/tiny-aya-safety-drift-map/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vincentni/tiny-aya-safety-drift-map/blob/main/notebooks/reproduce_tables.ipynb)

Cohere's Tiny Aya Global (3.35B) is asked questions about passages, half of which the passage cannot answer
(SQuAD 2.0). A probe on its layer-20 activations, trained on the model's own errors, flags the answers it is about
to get wrong; the model abstains where the probe fires and answers as before elsewhere. Lexsi AI Safety &
Alignment Hackathon 2026, Track 2b: Hallucination detection and abstention.

| On 1,000 SQuAD 2.0 dev questions, 500 of them unanswerable | Wrong answers | Wrong answers as read by a closed judge | Unanswerable questions answered, of 500 | Answerable questions still answered, of 500 | Of them with a gold answer |
|---|---|---|---|---|---|
| Base model | 560 (56.0%) | 505 (50.5%) | 485 | 497 | 425 |
| One instruction line in the prompt, no probe | 364 (36.4%) | 335 (33.5%) | 201 | 378 | 337 |
| **Probe on the activations as the gate** | 286 (28.6%) | 248 (24.8%) | 140 | 401 | 354 |
| LoRA fine-tune on the probe's 2,000 training questions | 174 (17.4%) | 155 (15.5%) | 70 | 431 | 396 |

*Wrong: an unanswerable question answered, or an answerable one answered incorrectly or abstained on; word match is the rule set in advance, and the judge's reading, the instruction line and the fine-tune were added after the main result.*

- **Problem**: the base model answers 485 of 500 questions its passage cannot answer.
- **Fix**: the probe gate takes wrong answers 560 → 286 (-27.4 points, 95% CI over articles -32.3 to -22.8) with no weight change, on a CPU; 401 of 500 answerable questions are still answered.
- **Baselines**: one instruction line gets 364, about 70% of that reduction; a LoRA fine-tune on the same 2,000 questions gets 174 (155 by the judge), better where the weights can be trained, and tested in distribution only. A second sample of questions repeats it: 575 → 315, the fine-tune 188.
- **Kits**: CuratorKIT gates the questions; AuditKIT generates the scored answers; CircuitKIT reads the same hidden states.
- **Limits**: English only; one probe, one fine-tune, one seed each; one prompt wording.

[Write-up](docs/article.md) | [Results Space](https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map) | [Models](https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-models) | [Datasets](https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-data) | [Notebook](notebooks/reproduce_tables.ipynb) | [Commands](docs/commands.md)

**Content warning:** the side checks use red-teaming prompts, and the datasets hold harmful requests and
answers. Nothing harmful is quoted in this repository; the results page withholds the text of answers judged harmful.

![A walk through the results page](docs/figures/explorer.gif)

*The [results Space](https://huggingface.co/spaces/vincentnkq/tiny-aya-safety-map) in 26 seconds.*

## Results

| On 1,000 SQuAD 2.0 dev questions, 500 of them unanswerable | Wrong answers | Wrong answers as read by a closed judge | Unanswerable questions answered, of 500 | Answerable questions still answered, of 500 | Of them with a gold answer |
|---|---|---|---|---|---|
| Abstaining on every question | 500 (50.0%) | 500 (50.0%) | 0 | 0 | 0 |
| Probe gate, else the instructed answer (exploratory) | 276 (27.6%) | 256 (25.6%) | 68 | 320 | 292 |

The fine-tune has fewer wrong answers than the probe gate (174 against 286; -11.2 points, 95% CI over articles -14.3 to -7.9; as the judge reads the answers 155 against 248, -9.3 points, 95% CI over articles -11.9 to -6.5) and keeps more answerable questions (431 against 401); it is the supervised reference: it costs a training run, changes the weights, was trained on labelled questions of the same dataset and format with the abstention sentence the word-match rule recognises, and is tested in distribution only. The probe leads here because it is the method the problem statement asks for, changes no weight and runs on a CPU. Labels used: the instruction line none; the probe the model's own errors on 2,000 train questions, its threshold set on train data; the fine-tune the gold answers of the same 2,000.

- **Not a text cue, and more than output confidence**: AUC against the model's error 0.8246 for the probe, 0.4978 for passage length, 0.5545 for TF-IDF; on the second sample the model's own output confidence 0.6951 at best, the probe 0.7853.
- **Three hackathon kits in the result**: CuratorKIT gates the questions, AuditKIT generates the scored answers, CircuitKIT reads the same hidden states (same probe decision on 200 of 200 prompts) and finds the same layer. The probe's fit, the baselines and the statistics are scikit-learn and scipy.
- **The limits**: English only; one probe, one fine-tune, one seed each, one prompt wording; word-match scoring (a closed judge's labels give 505 → 248); the errors it flags are mostly answered unanswerable questions.

More in the [write-up](docs/article.md): the served gate, where the gate pays, [the datasets](docs/article.md#sources-of-data) and [what does not carry](docs/question-pairs.md).

## Quick start

```bash
uv python install 3.12 && uv sync --frozen
uv run pytest -q -m "not smoke"                  # the offline tests
python space/build_data.py                       # the results page's data, from the public labels
python -m http.server --directory space          # the results page at http://localhost:8000/
uv run driftmap --help                           # the pipeline behind the numbers
```

## Licences and use

- Code: Apache-2.0, see `LICENSE`.
- Tiny Aya Global is cc-by-nc-4.0. Every weight derived from it here is under the same licence:
  non-commercial use only.
- Source datasets: BBQ, MBBQ and XSTest are cc-by-4.0, UnQover's templates apache-2.0, `aya_redteaming` and `aya_dataset` are Apache-2.0; the
  evaluation set built from them keeps those terms. Belebele and SQuAD 2.0 are cc-by-sa-4.0: no passage,
  question or gold answer of theirs is redistributed; the published rows are keyed by the datasets' own question ids.
- The adapters are research artifacts, published so the measurements can be repeated. They are
  evaluated only as described here and are not certified for deployment.
