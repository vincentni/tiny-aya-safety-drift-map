"""The data and the scoring behind the attribution fine-tune, as steps that run on a workstation.

Each module is a step with its own command line; inputs and outputs are folders and files named as arguments:

    python -I -m driftmap.attribution.build --bbq BBQ --benign sft_train/en.jsonl --out DATA
    python -m driftmap.attribution.translate --source DATA/test_en.jsonl --out DATA --english
    python -m driftmap.attribution.curator_report --data DATA --prompts PROMPTS --work SCRATCH \
        --out DATA/curator_report.json
    python -I -m driftmap.attribution.siblings --bbq BBQ --manifest DATA/manifest.json --test DATA/test_en.jsonl \
        --answers base=BASE.jsonl --answers fine_tuned=TUNED.jsonl --report DATA/curator_report.json
    python -I -m driftmap.attribution.free_form_rows --prompts PROMPTS --out FREE base=BASE.jsonl tuned=TUNED.jsonl
    python -m driftmap.attribution.free_form --out SCORED FREE/base.jsonl FREE/tuned.jsonl

``build`` draws the training rows and the English test items from BBQ; ``translate`` puts the test items into the
other languages and keeps the ones a blind check confirms; ``curator_report`` runs the CuratorKIT gate and the
dedup and leak checks over the training rows; ``siblings`` counts the held-out templates that share a stereotype
with a trained one; ``free_form_rows`` joins the free-form answers ``driftmap table choice`` sets aside with their
prompt rows; ``free_form`` reads which option an answer in free words asserts. ``translate`` and
``free_form`` call the closed model named by the environment and keep its replies in a cache beside their output,
so a second run of the same command makes no call.

The reading check asks both models questions from two other datasets, on a workstation CPU:

    python -I -m driftmap.attribution.reading --sources SOURCES --translation DATA/translation_manifest.json --out OUT
    python -m driftmap.attribution.local_generate --model MODEL_DIR --label REPO@REVISION \
        --manifest EVAL/manifest.json --input OUT/belebele_in.jsonl --output OUT/belebele_base.jsonl \
        --max-new-tokens 24 --batch-size 10 --workers 3 --threads 10
    python -I -m driftmap.attribution.reading_scores belebele --base OUT/belebele_base.jsonl \
        --fine-tuned OUT/belebele_ft.jsonl --out OUT
    python -I -m driftmap.attribution.reading_answers belebele --sources SOURCES --base OUT/belebele_base.jsonl \
        --fine-tuned OUT/belebele_ft.jsonl --out PUBLISHED

``reading`` draws the prompts from SQuAD 2.0 and Belebele; ``local_generate`` answers a prompt file with a local
checkpoint through the generate job's rendering and backend call, once per model and prompt file (24 new tokens
for Belebele, 48 for SQuAD 2.0); ``reading_scores`` scores the answers by rule, ``belebele`` or ``squad``;
``reading_answers`` writes the answers as rows that name each question by its id in the source dataset and hold
no text of it.

The question pairs in four languages come from MBBQ:

    python -I -m driftmap.attribution.mbbq_build --mbbq MBBQ --bbq BBQ --data DATA --out OUT
    python -I -m driftmap.attribution.mbbq_answers --prompts OUT/mbbq/prompts/benign --out PUBLISHED \
        base=BASE_ANSWERS attribution=TUNED_ANSWERS

``mbbq_build`` draws, from MBBQ at a pinned commit, items of question types the fine-tune never saw, the same
items in English, Spanish, Dutch and Turkish; the generate job answers them with ``--sets mbbq`` and
``driftmap table choice --sets mbbq`` scores them; ``mbbq_answers`` writes the items and the answers as rows that
name each prompt by its place in MBBQ and hold no text of it.

A side check on the evaluation prompts, from a copy of the responses dataset:

    python -I -m driftmap.attribution.answer_lengths RESPONSES_DIR OUT.json --model "base model=FOLDER" \
        --model "attribution fine-tune=FOLDER"

``answer_lengths`` counts, per model and prompt set, the rows, the mean answer length in tokens, the answers that
stopped at the token limit and the answers that are a trained abstention sentence; it writes no prompt or answer.
"""
