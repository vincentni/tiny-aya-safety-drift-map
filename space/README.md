---
title: Tiny Aya safety failure map
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
short_description: Where Tiny Aya Global gives harmful answers, by language
datasets:
  - vincentnkq/aya-redteaming-6lang-eval
  - vincentnkq/safety-drift-map-results
models:
  - CohereLabs/tiny-aya-global
  - vincentnkq/tiny-aya-global-drift-lora
---

# Tiny Aya safety failure map

A static results page for the study in <https://github.com/vincentni/tiny-aya-safety-drift-map>: where
Tiny Aya Global gives harmful answers on 600 red-teaming prompts in six languages, by language, harm
category and prompt condition, before and after a benign multilingual LoRA fine-tune.

**Content warning:** the study uses red-teaming prompts. This page shows prompt ids, languages, harm
categories and the judge's labels. It shows no prompt text and no model answer; those are in the
datasets, behind their own warnings.

## What is on the page

- The headline counts and the figures of the write-up.
- The failure map: harmful answers of the base model and the fine-tuned model for every language and
  harm category, with 95% Wilson intervals, filtered by condition, language and category.
- A prompt explorer: for each prompt id, the judge's labels for both models in both conditions, and
  whether the verdict changed.

No model runs here. The page reads `data.json`, which holds counts and labels only.

## Rebuild the data file

```bash
python space/build_data.py
```

The script downloads the published label files and tables, and writes `space/data.json`. With
`--eval-dir` and `--tables-dir` it reads local copies instead.

## Files

| File | Content |
| --- | --- |
| `index.html` | the page, with its styles |
| `app.js` | filters and tables, no dependencies |
| `data.json` | counts, intervals and labels, written by `build_data.py` |
| `figures/` | the figures, copied from `docs/figures/` by `build_data.py` |
