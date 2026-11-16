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
  - vincentnkq/tiny-aya-global-attribution-lora
  - vincentnkq/tiny-aya-global-drift-lora
---

# Tiny Aya Global: a probe on its activations as an abstention gate

A static results page for the study in <https://github.com/vincentni/tiny-aya-safety-drift-map>: Tiny Aya
Global is asked questions about passages, half of which the passage cannot answer (SQuAD 2.0), and a probe on
its layer-20 hidden state, trained on the model's own errors, makes it abstain where it fires. Below it, the
secondary evidence: who-did-it question pairs from BBQ and MBBQ before and after a LoRA fine-tune, and the side
checks of that fine-tune on 600 red-teaming prompts in six languages.

**Content warning:** the study uses red-teaming prompts. The tables and figures show counts only. The
replay explorer shows prompt text as it was written, and the stored answer of each system behind a
click. Some prompts are offensive. The text of answers judged harmful is not shown here; it is in
the responses dataset, under its own warning.

## What is on the page

- The headline numbers and the figures of the write-up.
- A probe as an abstention gate: on 1,000 SQuAD 2.0 questions, side by side, the base model, the base model
  with one instruction line in the prompt and no probe, the model that abstains where the probe fires, the
  probe's gate on top of the instructed answers (an exploratory computation) and abstaining on every question;
  the same answers as a closed judge reads them; a second sample of questions; the gate served inside
  generation; the probe beside the instruction line, a fine-tune and the model's own confidence in its answer,
  one row per system; the probe beside two scores that read only the text; risk against coverage by threshold; two
  probes fitted on question pairs, scored on the same questions. No text of SQuAD is shown or held in the
  data file.
- Secondary evidence. Who did it, when the text does not say: for each arm and language, how often a person is named on a
  question the text cannot answer, and twin accuracy, with 95% Wilson intervals, on 600 question pairs
  from MBBQ in four languages; the arms are the base model, the base model with the safety patch, a
  benign fine-tune with the same settings (the light fine-tune) and the attribution fine-tune. Beside
  it, in its own table and never pooled with it, BBQ's English test with two more prompt-level arms.
  One English question pair from BBQ with the stored answers of the base model and of the attribution
  fine-tune. No MBBQ text is shown.
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
has painted `samples.json`, which holds the one question pair and its stored answers, and
`explorer.json`, which holds the explorer's prompts and stored answers.

## Rebuild the data file

```bash
python space/build_data.py
```

To see the page from a clone, serve the folder over HTTP: `python space/build_data.py`, then
`python -m http.server --directory space`, and open <http://localhost:8000/>. Opened as a file the page
shows a notice instead, because a browser does not let a page fetch local files.

The script downloads the published label files and tables from the datasets `driftmap.toml` names,
and writes `space/data.json`. The answer-length counts need the responses dataset; when it cannot be
read they are left out and the page hides that table. With `--eval-dir`, `--tables-dir` and
`--responses-dir` the script reads local copies instead.

The probe is read from its tables in the results dataset (`tables/probe-squad.json` and, where they exist,
the other `probe-squad-*.json` tables beside it: the judge's reading, the served gate, the identity check, the
second sample, both samples together, the instruction line, the probe's gate on top of it, the fine-tune and
the confidence gates; `--choice-dir`
reads a local copy). The question pairs are read from the same three datasets: the two choice tables (the official items
in four languages, the English test) from the results dataset, the sample pair from the choice prompts
of the evaluation dataset and its stored answers from the responses dataset (`--choice-dir`, `--choice-eval-dir` and `--choice-responses-dir` read local
copies). The script writes the counts into `space/data.json` and the question pair into
`space/samples.json`; when a dataset cannot be read the page leaves that part out.

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
| `samples.json` | one question pair with its stored answers, written by `build_data.py` |
| `explorer.json` | prompts and stored answers for the explorer, written by `build_data.py`; not tracked |
| `figures/` | the figures, copied from `docs/figures/` by `build_data.py`; not tracked |
