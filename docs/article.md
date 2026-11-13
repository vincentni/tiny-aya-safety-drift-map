# A per-language map of safety drift in Tiny Aya Global, and whether a weight-space repair survives 4-bit quantization

Fine-tuning Tiny Aya Global on a harmless multilingual task and measuring, language by language, how much refusal behaviour it loses, how much a retraining-free repair gives back, and what is left after q4_k_m quantization.

**Tracks and problem statements**

- Track 2b: Repairing safety drift without retraining (main)
- Track 2b: Safety-aware compression (the quantized column)
- Track 2a: Multilingual and code-mixed safety (the per-language table)

> Status, Nov 13: the pipeline code and its tests are in the repo and the first GPU runs are about to start. Result cells below say "pending" until a run has produced them. I will resubmit as the columns fill in.

## Abstract

Safety drift after fine-tuning is usually reported as one aggregate number judged in English. Tiny Aya is a multilingual model, and its own paper says the authors are most concerned about safety disparities across languages, so an aggregate hides the part that matters.

I fine-tune Tiny Aya Global (3.35B) with LoRA on a benign instruction task in six languages (English, French, Spanish, Russian, Hindi, Arabic), then measure the safe-response rate on 100 red-teaming prompts per language from `CohereLabs/aya_redteaming`, before and after. Responses are judged by PolyGuard-Qwen, a multilingual safety classifier, instead of an English-only judge. The drifted model is then repaired in weight space with SafeTune (interpolation back toward the original weights, no retraining), and the repaired model is quantized to q4_k_m and measured again. Next to safety I report over-refusal on XSTest and task utility (chrF on held-out benign prompts), so a repair that just refuses everything, or one that throws the fine-tune away, shows up as such.

- Headline before → after: pending (pooled safe rate, baseline → drifted → repaired → repaired q4_k_m)
- Base model and method: Tiny Aya Global, LoRA SFT for the drift, SafeTune WiSE-FT interpolation for the repair, llama.cpp q4_k_m for the quantized column

## Problem and approach

Someone who adapts a small multilingual model to their own task wants to know three things: did the adaptation cost safety, in which languages, and can it be fixed without paying for training again. The repaired model is also likely to be served quantized, and quantization can undo a repair or hide a regression, so that column is part of the answer.

Questions:

1. How large is the drop in safe-response rate after benign fine-tuning, per language, and is it larger for Hindi and Arabic than for the European languages?
2. Does a weight-space repair recover it without losing the fine-tuned task (chrF within one point) and without raising over-refusal?
3. Does the repair hold after q4_k_m quantization?

## Hackathon repos used

- **CuratorKIT**: builds the benign training set. Schema gate, exact and MinHash dedup, provenance manifest. I add a held-out carve per language and a check that no training row is a near copy of an evaluation prompt.
- **SafeTune**: the repair. `WiseFTTrainer` interpolates the drifted weights toward Tiny Aya Global over a small alpha screen; the smallest alpha that passes both bars (safety within 5 points of baseline, chrF within 1 point of the drifted model) is kept.
- **AuditKIT**: generation and scoring for every column, chrF, and the paired comparison between columns.

Pipeline: CuratorKIT data → LoRA drift → AuditKIT generate → PolyGuard judge → SafeTune repair → llama.cpp quantize → generate and judge again → tables.

## Models and licenses

| Model | Role | License |
|---|---|---|
| `CohereLabs/tiny-aya-global` | base, and the reference the repair interpolates toward | cc-by-nc-4.0 |
| `ToxicityPrompts/PolyGuard-Qwen` | safety judge | see model card |

The fine-tuned (drifted) adapter is published for research only and is not served anywhere. The demo, if there is one, serves the repaired model.

## Datasets and benchmarks

| Source | Use | License |
|---|---|---|
| `CohereLabs/aya_redteaming` | 100 harmful prompts per language, stratified by global/local | apache-2.0 |
| `CohereLabs/aya_dataset` | benign training rows and 100 held-out rows per language | apache-2.0 |
| `Paul/XSTest` | 250 safe prompts for over-refusal (English) | cc-by-4.0 |

Sizes after filtering: 100 harmful and 100 held-out benign prompts per language, 212 training rows per language (1,272 in total), 250 XSTest prompts. Leak check: exact match after Unicode folding plus trigram Jaccard at 0.85 between every training row and every evaluation prompt, across languages. The benign pool lost 3,423 rows as repeats of an earlier training prompt and 364 more to CuratorKIT's dedup.

## Metrics

- Safe-response rate on harmful prompts, per language and pooled (higher is better), with Wilson intervals
- Over-refusal rate on XSTest safe prompts (lower is better)
- chrF on held-out benign prompts (higher is better); this is the "should not get worse" metric
- Differences between columns are paired by prompt: bootstrap 95% interval and an exact sign test

## Results

| Language | Baseline | Drifted | Repaired | Repaired q4_k_m |
|---|---|---|---|---|
| en, fr, es, ru, hi, ar | pending | pending | pending | pending |

## Limitations

- One seed, 100 prompts per language: per-language intervals are wide, pooled numbers are the ones to trust.
- The judge is a model. A spot check of a sample with a second judge is planned, to report agreement.
- XSTest is English only, so over-refusal is not measured per language.
- The training set is small after dedup; the amount of drift depends on the number of epochs, which is reported with the result.

## With more credits

More seeds, the remaining `aya_redteaming` languages, a second repair method for comparison, and a circuit-level look (CircuitKIT) at where the lost refusals live. Expected: tighter intervals and a per-language repair recommendation instead of one alpha for all.

## Artifacts

- Code: https://github.com/vincentni/tiny-aya-safety-drift-map
- Models: https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-models-6af750027e17226835562a9a
- Data: https://huggingface.co/collections/vincentnkq/tiny-aya-safety-drift-map-data-6af75002f6e0994706b717d2

## Reproduction

    uv sync
    uv run pytest -q -m "not smoke"

Every GPU step is a Hugging Face Job launched through the `driftmap` CLI, with a cost ledger and a dry-run mode. Commands for each column are added here as the runs complete.

Work done outside the hackathon window: none. Everything in the repo was written after Nov 1.

## Team

Vincent Ni ([@vincentnkq](https://huggingface.co/vincentnkq)), solo.
