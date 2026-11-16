---
title: Tiny Aya abstains where a probe fires
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
short_description: A probe as an abstention gate on SQuAD 2.0
datasets:
  - vincentnkq/aya-redteaming-6lang-eval
  - vincentnkq/safety-drift-map-results
  - vincentnkq/safety-drift-map-responses
models:
  - CohereLabs/tiny-aya-global
  - vincentnkq/tiny-aya-global-squad-abstain-lora
  - vincentnkq/tiny-aya-global-attribution-lora
  - vincentnkq/tiny-aya-global-drift-lora
  - vincentnkq/tiny-aya-global-drift-lora-nopre
  - vincentnkq/tiny-aya-global-repaired-nopre
  - vincentnkq/tiny-aya-global-drift-GGUF
---

# Tiny Aya abstains where a probe fires

A static results page for the study in <https://github.com/vincentni/tiny-aya-safety-drift-map>. Tiny Aya Global is
asked questions about passages, half of which the passage cannot answer (SQuAD 2.0); a probe on its layer-20
hidden state makes it abstain where it fires. One instruction line in the prompt and a LoRA fine-tune on the
probe's own training questions are set beside it. No model runs here.

**Content warning:** the secondary evidence uses red-teaming prompts, some of them offensive. The replay explorer
shows prompt text and, behind a click, stored answers; the text of answers judged harmful is not shown.

## What is on the page

- **The probe as an abstention gate**: the main table (base model, instruction line, probe, the LoRA fine-tune, abstaining on every question), the
  same answers as a closed judge reads them, a second sample of questions, one row per system beside a fine-tune
  on question pairs and the model's own confidence, two text-only scores, and risk against coverage. No text of
  SQuAD is shown or held in the data file.
- **Secondary evidence**: who-did-it question pairs from BBQ and MBBQ before and after a fine-tune (no MBBQ text
  is shown), and that fine-tune's side checks on 600 red-teaming prompts: the failure map and the replay explorer.

## Rebuild the data file

```bash
python space/build_data.py
python -m http.server --directory space
```

The script downloads the published tables and label files from the datasets `driftmap.toml` names and writes
`data.json`, `samples.json`, `explorer.json` and `figures/`; `--tables-dir`, `--eval-dir`, `--responses-dir` and
the three `--choice-*` options read local copies. A part whose dataset cannot be read is left out of the page.
`driftmap publish space` refuses an explorer file that holds the text of harmful answers or is older than the
data file.
