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
category and prompt condition, before and after two LoRA fine-tunes on the same benign multilingual rows.

**Content warning:** the study uses red-teaming prompts. This page shows counts only: no prompt text,
no model answer and no per-prompt row. Those are in the datasets, behind their own warnings.

## What is on the page

- The headline numbers and the figures of the write-up.
- What the light fine-tune did and did not do: harmful answers among complete answers and among
  answers cut off at the token limit.
- The overfit recipe: harmful answers by language of the base model and of both fine-tunes, with no
  safety preamble.
- A targeted system-prompt patch and the script stress test (prompts in Latin script or mixed with
  English), each shown once its label files are in the evaluation dataset.
- The failure map: harmful answers of every measured model for every language and harm category,
  with 95% Wilson intervals, filtered by condition, language and category.

No model runs here. The page reads `data.json`, which holds counts and intervals only.

## Rebuild the data file

```bash
python space/build_data.py
```

The script downloads the published label files and tables, and writes `space/data.json`. The
answer-length counts need the responses dataset, which sits behind an access gate; without access
they are left out and the page hides that table. With `--eval-dir`, `--tables-dir` and
`--responses-dir` the script reads local copies instead.

## Files

| File | Content |
| --- | --- |
| `index.html` | the page, with its styles |
| `app.js` | filters and tables, no dependencies |
| `data.json` | counts and intervals, written by `build_data.py` |
| `figures/` | the figures, copied from `docs/figures/` by `build_data.py` |
