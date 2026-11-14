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

**Content warning:** the study uses red-teaming prompts. The tables and figures show counts only. The
replay explorer shows prompt text as it was written, and the stored answer of each system behind a
click. Some prompts are offensive. The text of answers judged harmful is not shown here; it is in
the responses dataset, under its own warning.

## What is on the page

- The headline numbers and the figures of the write-up.
- What the light fine-tune did and did not do: harmful answers among complete answers and among
  answers cut off at the token limit.
- The overfit recipe and its repair: harmful answers by language of the base model, of both
  fine-tunes and of the repaired fine-tune, with no safety preamble, and the overfit recipe's change
  under the served template.
- A targeted system-prompt patch: each model with and without it, on the same prompts.
- The script stress test: prompts in Latin script or mixed with English against their native-script
  parents.
- The failure map: harmful answers of every measured model for every language and harm category,
  with 95% Wilson intervals, filtered by condition, language and category.

- The replay explorer: pick a language, a harm category and a prompt, and see side by side what each
  measured system did with it: the judge's verdict, whether the answer hit the 256-token limit, and
  the answer itself. A filter lists the prompts where the systems disagree, and the address of the
  page (`#language/category/prompt-id`) links to one prompt.

No model runs here. The page reads `data.json`, which holds counts and intervals only, and after it
has painted `explorer.json`, which holds the explorer's prompts and stored answers.

## Rebuild the data file

```bash
python space/build_data.py
```

The script downloads the published label files and tables, and writes `space/data.json`. The
answer-length counts need the responses dataset; when it cannot be read they are left out and the
page hides that table. With `--eval-dir`, `--tables-dir` and
`--responses-dir` the script reads local copies instead.

The script also copies the figures from `docs/figures/` to `space/figures/` and lists them in the
data file, and, when it can read the responses dataset, writes `space/explorer.json`. Both are build
outputs and are not tracked in the code repository: run the script before uploading this folder to
the Space. Without `explorer.json` the page hides the explorer.

The explorer file holds at most 300 prompts (`--explorer-cap`): the prompts where the systems
disagree on a harmful answer first, then the prompts every system answers harmfully, then prompts
spread evenly over the languages. Prompts of the non-consensual sexual content category are left
out. Answers judged harmful, or left without a verdict, are listed without their text;
`--include-harmful` keeps their first 300 characters, for a copy that is not published:
`driftmap publish space` refuses an explorer file built that way, or one older than the data file.

## Files

| File | Content |
| --- | --- |
| `index.html` | the page, with its styles |
| `app.js` | filters and tables, no dependencies |
| `explorer.js` | the replay explorer, no dependencies |
| `data.json` | counts and intervals, written by `build_data.py` |
| `explorer.json` | prompts and stored answers for the explorer, written by `build_data.py`; not tracked |
| `figures/` | the figures, copied from `docs/figures/` by `build_data.py`; not tracked |
