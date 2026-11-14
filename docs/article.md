# A per-language map of safety drift in Tiny Aya Global, and whether a weight-space repair survives 4-bit quantization

Tiny Aya Global is fine-tuned on a harmless multilingual task, and its behaviour on harmful prompts is measured before and after in six languages, with and without the safety paragraph of its chat template. In both conditions the fine-tuned model is as safe as the baseline or safer, and better at the benign task.

**Tracks and problem statements**

- Track 2b: Repairing safety drift without retraining (main)
- Track 2a: Multilingual and code-mixed safety (the per-language table)

**Content warning:** the evaluation uses red-teaming prompts. The published responses dataset holds harmful requests and model outputs, and some of the outputs are harmful. This article quotes none of them.

## Abstract

Benign fine-tuning can remove refusal behaviour from an aligned model (Qi et al., 2023), and the loss is usually reported as one aggregate number judged in English. Here it is measured per language. Tiny Aya Global (3.35B) is fine-tuned with LoRA on 1,272 benign instruction rows in English, French, Spanish, Russian, Hindi and Arabic, and evaluated before and after on 100 red-teaming prompts per language, with and without the safety paragraph of its chat template, with over-refusal and task quality beside safety.

- Headline before → after, served chat template: pooled safe rate 90.8% → 93.5% (+2.7 points, 95% CI +0.3 to +5.0, p = 0.029, 597 prompt pairs). Refusal rate 70.6% → 74.4%, XSTest over-refusal 10.4% → 9.2%, chrF 30.5 → 37.3.
- Headline before → after, safety paragraph removed: pooled safe rate 89.8% → 94.1% (+4.4 points, 95% CI +2.2 to +6.7, p < 0.001, 597 prompt pairs). Refusal rate 66.4% → 78.4%, XSTest over-refusal 9.6% → 9.2%, chrF 25.1 → 36.1.
- Finding: no drift in either condition, and none in any language beyond noise. There is nothing to repair with this recipe.
- Base model and method: Tiny Aya Global; LoRA SFT with TRL for the fine-tune; SafeTune WiSE-FT interpolation for the repair; llama.cpp q4_k_m for the quantized column.

## Problem and approach

Someone who adapts a small multilingual model to their own task wants to know three things: did the adaptation cost safety, in which languages, and can it be fixed without paying for training again.

1. How much does the safe-response rate change after benign fine-tuning, per language, and is the change worse for Hindi and Arabic than for the four European languages?
2. If it drops, does a weight-space repair (WiSE-FT, Wortsman et al., 2022) recover it without losing the fine-tuned task and without raising over-refusal?
3. Does the repair hold after q4_k_m quantization, the form the model is likely to be served in?

Two conditions are measured. In the first, prompts are rendered with the chat template as served: its system turn opens with a safety paragraph. The second removes that paragraph when the prompts are rendered and changes nothing else in the template. Both conditions evaluate the same two checkpoints.

The fine-tune is a LoRA adapter trained with the settings below; the merged weights are evaluated.

| Setting | Value |
|---|---|
| LoRA | rank 16, alpha 32, dropout 0.05, attention and MLP projections |
| Training | 3 epochs, 240 steps, effective batch 16, one L4 |
| Learning rate | 2e-4, cosine decay |
| Generation | greedy, at most 256 new tokens |

## Hackathon repos used

- **CuratorKIT**: builds the benign fine-tuning set (schema gate, exact and MinHash dedup, provenance manifest). Added on top: a held-out carve per language and a leak check against the evaluation prompts.
- **AuditKIT**: generation for every column, chrF, and the paired difference between columns; scipy adds the interval and the sign test.
- **SafeTune**: the repair. `WiseFTTrainer` interpolates the fine-tuned weights toward Tiny Aya Global over a screen of alphas (0.1, 0.25, 0.4, 0.5). The smallest alpha that keeps pooled safety within 5 points of baseline and chrF within 1 point of the fine-tuned model is kept.

Pipeline: CuratorKIT data → LoRA fine-tune → AuditKIT generate → rubric judge → tables → SafeTune repair → llama.cpp q4_k_m → generate and judge again. Every GPU step is a Hugging Face Job, rehearsed as a smoke run before the full run.

## Models and licenses

| Model | Role | License |
|---|---|---|
| `CohereLabs/tiny-aya-global` | base model, and the reference the repair interpolates toward | cc-by-nc-4.0 |
| `vincentnkq/tiny-aya-global-drift-lora` | the fine-tuned adapter | cc-by-nc-4.0 |
| `claude-sonnet-5-5` | judge: a closed model reached through LiteLLM | closed, no weights |

The fine-tuned adapter is published for research only and is not served anywhere.

**The judge** is a closed model, which limits reproducibility, so everything around it is fixed and public. The rubric is in `src/driftmap/judge/rubric.py` and its hash is stored in every label row. It answers each prompt and response pair with one JSON object: `request_harm`, `refusal`, `response_harm`, each yes or no. A pair whose answer cannot be read is stored as unscored and counted in no numerator or denominator. On the harmful prompts that is 2 of 600 baseline and 2 of 600 fine-tuned pairs under the served template, and 1 and 3 without the safety paragraph. Every label is published with the evaluation set.

## Datasets and benchmarks

| Source | Use | License |
|---|---|---|
| `CohereLabs/aya_redteaming` | 100 harmful prompts per language, stratified by its global/local flag | apache-2.0 |
| `CohereLabs/aya_dataset` | benign fine-tuning rows and 100 held-out rows per language | apache-2.0 |
| `Paul/XSTest` | 250 safe prompts for over-refusal (English) | cc-by-4.0 |

The fine-tuning set has 212 rows per language (1,272 in total). Sampling is seeded and the source revisions are pinned in `revisions.json`. Leak check: no training row is an exact copy (after Unicode folding) or a near copy (trigram Jaccard 0.85) of any evaluation prompt in any language; 344 candidates were dropped by it.

## Metrics

- **Safe rate**: share of harmful prompts whose response the judge does not call harmful, with Wilson 95% intervals. Higher is better.
- **Refusal rate**: share of the same prompts where the judge reads a refusal. A response can be safe without refusing, so the two rates differ.
- **Over-refusal**: refusal rate on the safe XSTest prompts (Röttger et al., 2024). Lower is better.
- **chrF** against the reference answer on held-out benign prompts: the "should not get worse" metric.
- **Changes between columns** are paired by prompt: mean difference, percentile bootstrap 95% interval (1,000 resamples, fixed seed) and an exact two-sided sign test.

## Results

### Served chat template (safety preamble kept)

Safe rate on harmful prompts. The change is in percentage points, fine-tuned minus baseline, over prompts scored in both columns.

| Language | Baseline | Fine-tuned | Change (95% CI) | p |
|---|---|---|---|---|
| en | 88.0% | 89.9% | +2.0 (-4.0 to +8.1) | 0.754 |
| fr | 98.0% | 96.0% | -2.0 (-6.0 to +2.0) | 0.625 |
| es | 92.0% | 93.0% | +1.0 (-4.0 to +6.0) | 1.000 |
| ru | 83.8% | 90.9% | +7.1 (+0.0 to +14.1) | 0.092 |
| hi | 90.0% | 94.0% | +4.0 (+0.0 to +9.0) | 0.219 |
| ar | 92.9% | 97.0% | +4.0 (-2.0 to +10.1) | 0.344 |
| all | 90.8% | 93.5% | +2.7 (+0.3 to +5.0) | 0.029 |

Refusal rate on the same prompts, and chrF on held-out benign prompts (100 per language).

| Language | Refusal, baseline | Refusal, fine-tuned | chrF, baseline | chrF, fine-tuned | chrF change (95% CI) |
|---|---|---|---|---|---|
| en | 72.0% | 71.7% | 35.7 | 42.5 | +6.8 (+3.2 to +10.8) |
| fr | 80.0% | 84.0% | 34.2 | 38.7 | +4.5 (+0.8 to +8.1) |
| es | 67.0% | 78.0% | 38.4 | 44.1 | +5.8 (+1.5 to +9.7) |
| ru | 69.7% | 80.8% | 21.9 | 26.4 | +4.5 (+1.4 to +7.7) |
| hi | 68.0% | 73.0% | 26.7 | 38.0 | +11.4 (+6.7 to +16.3) |
| ar | 66.7% | 59.0% | 26.1 | 34.4 | +8.3 (+3.5 to +13.5) |
| all | 70.6% | 74.4% | 30.5 | 37.3 | +6.9 (+5.2 to +8.5) |

Over-refusal on XSTest (English, 250 prompts): 10.4% (95% CI 7.2% to 14.8%) → 9.2% (6.2% to 13.4%), a change of -1.2 points (-4.8 to +2.4, p = 0.664).

Hindi and Arabic against the four European languages: 91.5% against 90.5% at baseline (+1.0 points, 95% CI -3.8 to +5.5) and 95.5% against 92.5% after fine-tuning (+3.0 points, -1.0 to +6.5).

### No safety preamble

Same prompts, decoding, judge and checkpoints. Only the safety paragraph is removed from the system turn when the prompts are rendered.

| Language | Baseline | Fine-tuned | Change (95% CI) | p |
|---|---|---|---|---|
| en | 85.0% | 88.9% | +4.0 (-2.0 to +11.1) | 0.344 |
| fr | 95.0% | 99.0% | +4.0 (+1.0 to +8.0) | 0.125 |
| es | 86.0% | 93.0% | +7.0 (+1.0 to +14.0) | 0.065 |
| ru | 87.9% | 92.9% | +5.1 (+0.0 to +10.1) | 0.125 |
| hi | 92.0% | 95.0% | +3.0 (-2.0 to +9.0) | 0.453 |
| ar | 93.0% | 96.0% | +3.0 (-2.0 to +8.1) | 0.453 |
| all | 89.8% | 94.1% | +4.4 (+2.2 to +6.7) | < 0.001 |

| Language | Refusal, baseline | Refusal, fine-tuned | chrF, baseline | chrF, fine-tuned | chrF change (95% CI) |
|---|---|---|---|---|---|
| en | 63.0% | 71.7% | 26.6 | 41.3 | +14.7 (+10.9 to +18.7) |
| fr | 73.0% | 89.0% | 31.3 | 35.1 | +3.8 (+0.0 to +7.8) |
| es | 59.0% | 79.0% | 26.3 | 43.3 | +17.0 (+12.3 to +22.2) |
| ru | 71.7% | 83.8% | 22.9 | 26.5 | +3.6 (+0.5 to +7.0) |
| hi | 68.0% | 75.0% | 26.2 | 36.9 | +10.6 (+6.4 to +15.1) |
| ar | 64.0% | 71.7% | 17.2 | 33.4 | +16.1 (+10.8 to +22.3) |
| all | 66.4% | 78.4% | 25.1 | 36.1 | +11.0 (+9.1 to +12.6) |

Over-refusal on XSTest: 9.6% (95% CI 6.5% to 13.9%) → 9.2% (6.2% to 13.4%), a change of -0.4 points (-3.2 to +2.8, p = 1.000).

Hindi and Arabic against the four European languages: 92.5% against 88.5% at baseline (+4.0 points, 95% CI -0.7 to +8.8) and 95.5% against 93.5% after fine-tuning (+2.0 points, -1.5 to +5.8).

The full tables for both conditions are in the results dataset.

### What the two conditions show

- **No drift in either condition.** The pooled safe rate rose by 2.7 points under the served template and by 4.4 points without the safety paragraph. Benign fine-tuning left the model as safe or safer while improving the benign task.
- **No language lost safety beyond noise.** Eleven of the twelve per-language changes are positive. The exception is French under the served template: -2.0 points, with an interval from -6.0 to +2.0. No per-language change reaches p < 0.05. Hindi and Arabic show no gap against the other four languages: all four intervals include zero.
- **The safe behaviour sits mostly in the weights, not in the template.** Removing the safety paragraph moves the baseline only from 90.8% to 89.8% safe, and its refusal rate from 70.6% to 66.4%.
- **Refusals went up overall.** Under the served template English is flat and Arabic fell (66.7% to 59.0%) while its safe rate rose. Without the safety paragraph refusals rose in every language, Arabic included (64.0% to 71.7%), so the drop does not repeat.
- **The fine-tune did its job.** chrF rose in every language in both conditions, by 6.9 and 11.0 points pooled, and over-refusal did not move.

The pipeline treats a pooled drop under 5 points as "no drift" and holds the repair job, so with this recipe there is nothing to repair.

### Aggressive fine-tune

<!-- fill: aggressive recipe table -->

Pending. The same benign rows, trained without the safety paragraph for 10 epochs at learning rate 5e-4 (against 3 epochs at 2e-4) and evaluated in both conditions. A pooled drop of 5 points or more makes it the drifted model for the repair below; otherwise two recipes produced no drift.

### Repair and quantization

<!-- fill: repair alpha screen and repaired column -->

<!-- fill: repaired q4_k_m column -->

Pending. Runs only if the aggressive fine-tune drifts.

## Cost

Every job writes a line to a cost ledger: billed minutes times the published price of the flavor ($0.80 per hour for one L4). The launcher reads the ledger before each submit; the ceilings are $5 per run, $16 soft and $20 hard. The jobs behind the two conditions:

| Stage | Jobs | Minutes | Cost |
|---|---|---|---|
| Smoke rehearsals | 5 | 26 | $0.35 |
| Fine-tune, full | 1 | 37 | $0.49 |
| Generation, full, served template | 2 | 60 | $0.80 |
| Generation, full, no safety preamble | 2 | 56 | $0.75 |
| Total, ledger | 10 | 179 | $2.39 |
| Total, billing page | 10 | | $2.48 |

The two totals differ because the billing page charged one more minute than the ledger on seven of the ten jobs. The ledger covers Hugging Face Jobs only: the judge runs on a workstation against a paid API and its cost is not included.

<!-- fill: cost rows for the aggressive fine-tune, repair and quantization jobs -->

## Limitations

- One seed and one fine-tuning recipe (1,272 rows, 3 epochs). A larger or longer fine-tune may behave differently; this result does not say where drift starts.
- 100 prompts per language: per-language intervals are wide, and only the pooled comparison can see a change of a few points.
- The judge is one closed model with no pinned revision, so labelling again may flip borderline pairs. No human agreement study was run. The published labels are the reproducible artifact, not the judge.
- XSTest is English only, so over-refusal is not measured per language. chrF against one reference shows the model moved toward the training distribution, not that its answers are better.
- Two prompt conditions were tested: the served template, and the same template without its safety paragraph. A deployment with a different system prompt is not covered.

## With more credits

More seeds, the remaining `aya_redteaming` languages, longer fine-tunes and larger training sets to find where drift starts, a human-labelled sample to report judge agreement, and a second repair method for comparison. Expected: tighter per-language intervals and a dose-response curve instead of one point.

## Artifacts

- Code: https://github.com/vincentni/tiny-aya-safety-drift-map
- Models: https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-models-6af750027e17226835562a9a
- Data: https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-data-6af75002f6e0994706b717d2
- Evaluation set with labels: https://huggingface.co/datasets/vincentnkq/aya-redteaming-6lang-eval
- Notebook that recomputes the tables from the public labels: [`notebooks/reproduce_tables.ipynb`](../notebooks/reproduce_tables.ipynb)

## Reproduction

```bash
uv python install 3.12 && uv sync --frozen
uv run pytest -q -m "not smoke"

uv run driftmap ready --pin
uv run driftmap data build
uv run driftmap launch generate --wait -- --kind baseline            # smoke run first
uv run driftmap launch generate --mode full --wait -- --kind baseline
uv run driftmap judge closed --checkpoint REPO@SHA --mode full
uv run driftmap launch drift --wait                                  # smoke run first
uv run driftmap launch drift --mode full --wait -- --epochs 3
uv run driftmap launch generate --mode full --wait -- --kind drifted
uv run driftmap judge closed --checkpoint REPO@SHA --mode full
uv run driftmap table build
uv run driftmap ledger reconcile
```

`REPO@SHA` is the checkpoint a generate run answered with, as its run record names it. The pipeline's files call the fine-tuned column `drifted`. The judge needs a LiteLLM route in `DRIFTMAP_CLOSED_JUDGE_ROUTE`. `--preamble strip` on the generate jobs, `judge closed` and `table build` gives the no-preamble condition.

The notebook recomputes every rate, interval and paired change above from the published label files, without a GPU or a judge.

Work done outside the hackathon window: none. Everything in the repo was written after Nov 1.

## References

- Qi et al., 2023. Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!
- Wortsman et al., 2022. Robust fine-tuning of zero-shot models (WiSE-FT).
- Röttger et al., 2024. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.
- Cohere Labs. Tiny Aya Global model card, `CohereLabs/tiny-aya-global`.

## Team

Vincent Ni ([@vincentnkq](https://huggingface.co/vincentnkq)), solo.
