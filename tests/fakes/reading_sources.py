"""Files in the shape of SQuAD 2.0 and Belebele, small and made up: no text of the datasets is in them."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from driftmap.attribution import build, reading

CODES = {"en": "eng_Latn", "hi": "hin_Deva", "ar": "arb_Arab", "ru": "rus_Cyrl"}
CLOSING = {"en": build.LETTER_INSTRUCTION, "hi": "closing line hi", "ar": "closing line ar", "ru": "closing line ru"}


def squad_rows():
    """Thirty articles of eight short passages and one long one; each passage has a question of either kind."""
    rows = []
    for article in range(30):
        for passage in range(9):
            words = 300 if passage == 8 else 40
            context = " ".join([f"article{article}", f"passage{passage}", *["word"] * words])
            for kind in ("answerable", "unanswerable"):
                answers = [f"answer {article}-{passage}"] * 2 if kind == "answerable" else []
                rows.append(
                    {
                        "id": f"{kind[:2]}{article:02d}{passage}",
                        "title": f"Article {article}",
                        "context": context,
                        "question": f"What does passage {passage} of article {article} say, {kind}?",
                        "answers": {"text": answers, "answer_start": [0] * len(answers)},
                    }
                )
    return rows


def belebele_rows(code):
    """Seventy passages with two questions each; the answer key is the same in every language."""
    return [
        {
            "link": f"https://example.org/passage/{passage}",
            "question_number": question,
            "flores_passage": f"{code} passage {passage}",
            "question": f"{code} question {question} on passage {passage}",
            "correct_answer_num": str((passage + question) % 4 + 1),
        }
        | {f"mc_answer{place}": f"{code} option {place} of {passage}.{question}" for place in range(1, 5)}
        for passage in range(70)
        for question in (1, 2)
    ]


def write(tmp_path, monkeypatch):
    """A folder of the files with their sha256 recorded, and the closing lines in a translation manifest."""
    folder = tmp_path / "sources"
    path = folder / reading.SQUAD_FOLDER / reading.SQUAD_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(squad_rows()), path)
    monkeypatch.setattr(reading, "SQUAD_FILES", {reading.SQUAD_FILE: hashlib.sha256(path.read_bytes()).hexdigest()})
    recorded = {}
    for code, name in CODES.items():
        path = folder / reading.BELEBELE_FOLDER / "data" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in belebele_rows(code)), encoding="utf-8")
        recorded[f"data/{name}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(reading, "BELEBELE_FILES", recorded)
    fixed = {
        code: {"fixed_strings": {build.LETTER_INSTRUCTION: line}} for code, line in CLOSING.items() if code != "en"
    }
    (tmp_path / "translation_manifest.json").write_text(json.dumps({"languages": fixed}), encoding="utf-8")
    return folder
