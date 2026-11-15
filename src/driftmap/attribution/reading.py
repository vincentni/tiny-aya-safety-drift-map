"""Build the prompts of the reading check on two other datasets, SQuAD 2.0 and Belebele.

    python -I -m driftmap.attribution.reading --sources SOURCES --translation DATA/translation_manifest.json --out OUT

``--sources`` is a folder for the files of the two datasets: the ones missing there are downloaded at the revisions
pinned here, and all are checked against their recorded sha256; they are read as data. Both datasets are licensed
CC BY-SA 4.0 and no file of them is kept in this repository. The closing line of the Belebele prompts in Hindi,
Arabic and Russian is the one the translation step wrote. The draws are seeded: the same inputs give the same
files, byte for byte.

squad_in.jsonl: 100 unanswerable and 100 answerable SQuAD 2.0 validation questions, each on its own passage of
under 250 words, the articles taken in turn; free-form prompt as in the test set.
belebele_in.jsonl: 60 Belebele questions, the same ones in English, Hindi, Arabic and Russian, each from its own
passage; four lettered options with the correct one at A, B, C and D equally often, the same order in every
language; closing line in the language's own wording.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from driftmap.attribution import pinned
from driftmap.attribution.build import FREE_INSTRUCTION as FREE
from driftmap.attribution.build import LETTER_INSTRUCTION, SEED

HUB = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{name}"
LICENCE = "CC BY-SA 4.0"
SQUAD = "rajpurkar/squad_v2"
SQUAD_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"
SQUAD_FOLDER = "squad_v2"
SQUAD_FILE = "squad_v2/validation-00000-of-00001.parquet"
SQUAD_FILES = {SQUAD_FILE: "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb"}
BELEBELE = "facebook/belebele"
BELEBELE_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BELEBELE_FOLDER = "belebele"
BELEBELE_FILES = {
    "data/eng_Latn.jsonl": "15af884c2b5994fdc71199e79d553a1864b08ce1b31894aa87f6be925a064a31",
    "data/hin_Deva.jsonl": "41d34aab7f8fc4d57dc5d61fb36c3ce6ce230fe8e1c156c4d3cff0ba4710ed8f",
    "data/arb_Arab.jsonl": "a2282b0974826172b1ce76a5d9e362091cfc8a7a25023df2d011f88f588b56e0",
    "data/rus_Cyrl.jsonl": "e8b81853605b97e993af07a34ad2c0a48ccd8dab80956b0650a161e8fea00de5",
}
PER_CONDITION = 100
MAX_WORDS = 250
ITEMS = 60
LETTERS = "ABCD"
LANGUAGES = {"en": "eng_Latn", "hi": "hin_Deva", "ar": "arb_Arab", "ru": "rus_Cyrl"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {path.name}")


def _fetch(sources: Path, repository: str, revision: str, folder: str, files: dict[str, str], get) -> dict[str, str]:
    def url(name: str) -> str:
        return HUB.format(repository=repository, revision=revision, name=name)

    return pinned.fetch(sources / folder, files, url, f"{repository} at {revision[:12]}", get)


def fetch_belebele(sources: Path, get: Callable[[str], bytes] = pinned.download) -> dict[str, str]:
    """The sha256 of the Belebele files under ``sources``, in their folder; missing files are downloaded."""
    return _fetch(sources, BELEBELE, BELEBELE_REVISION, BELEBELE_FOLDER, BELEBELE_FILES, get)


def fetch(sources: Path, get: Callable[[str], bytes] = pinned.download) -> dict[str, dict[str, str]]:
    """The sha256 of the files of the two datasets under ``sources``, a folder each; missing files are downloaded."""
    squad_files = _fetch(sources, SQUAD, SQUAD_REVISION, SQUAD_FOLDER, SQUAD_FILES, get)
    return {SQUAD_FOLDER: squad_files, BELEBELE_FOLDER: fetch_belebele(sources, get)}


def squad(source: Path, out: Path) -> None:
    """Write squad_in.jsonl under ``out`` from the validation file ``source``."""
    rng = random.Random(f"{SEED}-squad")
    rows = sorted(pq.read_table(source).to_pylist(), key=lambda row: row["id"])
    short = [row for row in rows if len(row["context"].split()) < MAX_WORDS]
    titles = sorted({row["title"] for row in short})
    rng.shuffle(titles)
    used, drawn = set(), {}
    for condition in ("unanswerable", "answerable"):
        pool = defaultdict(list)
        for row in short:
            if (len(row["answers"]["text"]) == 0) == (condition == "unanswerable"):
                pool[row["title"]].append(row)
        picked = []
        while len(picked) < PER_CONDITION:
            before = len(picked)
            for title in titles:
                left = [row for row in pool[title] if row["context"] not in used]
                if left and len(picked) < PER_CONDITION:
                    row = rng.choice(left)
                    used.add(row["context"])
                    picked.append(row)
            if len(picked) == before:
                raise SystemExit(f"only {len(picked)} {condition} questions on unused passages")
        drawn[condition] = picked
    rows_out = []
    for pair in zip(drawn["unanswerable"], drawn["answerable"], strict=True):
        for condition, row in zip(("unanswerable", "answerable"), pair, strict=True):
            rows_out.append(
                {
                    "id": f"squad.{row['id']}",
                    "condition": condition,
                    "context": row["context"],
                    "question": row["question"],
                    "gold_answers": sorted(set(row["answers"]["text"])),
                    "title": row["title"],
                    "prompt": "\n".join([row["context"], row["question"], FREE]),
                    "set": "benign",
                    "language": "en",
                }
            )
    write_jsonl(out / "squad_in.jsonl", rows_out)
    words = sorted(len(row["context"].split()) for row in rows_out)
    print(
        "   validation questions",
        len(rows),
        "| on passages under",
        MAX_WORDS,
        "words:",
        len(short),
        "| articles:",
        len(titles),
        "| articles drawn from:",
        len({row["title"] for row in rows_out}),
        "| distinct passages:",
        len({row["context"] for row in rows_out}),
        "| passage words median",
        words[len(words) // 2],
        "max",
        words[-1],
    )
    print("   questions per article:", dict(Counter(Counter(row["title"] for row in rows_out).values())))


def belebele(source: Path, closing: dict[str, str], out: Path) -> None:
    """Write belebele_in.jsonl under ``out`` from the language files under ``source``; ``closing`` holds the last
    line of a prompt per language."""
    rng = random.Random(f"{SEED}-belebele")
    files = {
        code: {(row["link"], row["question_number"]): row for row in read_jsonl(source / f"{name}.jsonl")}
        for code, name in LANGUAGES.items()
    }
    keys = sorted(set.intersection(*(set(rows) for rows in files.values())))
    same = [key for key in keys if len({files[code][key]["correct_answer_num"] for code in LANGUAGES}) == 1]
    by_link = defaultdict(list)
    for key in same:
        by_link[key[0]].append(key)
    links = rng.sample(sorted(by_link), ITEMS)
    items = [rng.choice(by_link[link]) for link in links]
    rows_out = []
    for number, key in enumerate(items):
        correct = int(files["en"][key]["correct_answer_num"]) - 1
        rest = [index for index in range(4) if index != correct]
        rng.shuffle(rest)
        place = number % 4
        placed = rest[:place] + [correct] + rest[place:]
        for code in LANGUAGES:
            row = files[code][key]
            texts = [row[f"mc_answer{index + 1}"] for index in placed]
            lines = [row["flores_passage"], row["question"]]
            lines += [f"{letter}. {text}" for letter, text in zip(LETTERS, texts, strict=True)]
            lines.append(closing[code])
            rows_out.append(
                {
                    "id": f"belebele.{code}.{number:02d}",
                    "item": f"{key[0]}#{key[1]}",
                    "language": code,
                    "prompt": "\n".join(lines),
                    "passage": row["flores_passage"],
                    "question": row["question"],
                    "options": [{"letter": letter, "text": text} for letter, text in zip(LETTERS, texts, strict=True)],
                    "letter_correct": LETTERS[place],
                    "set": "benign",
                }
            )
    write_jsonl(out / "belebele_in.jsonl", rows_out)
    print(
        "   questions in all four languages:",
        len(keys),
        "| with one answer key across them:",
        len(same),
        "| passages:",
        len(by_link),
        "| drawn:",
        len(items),
        "from",
        len(set(links)),
        "passages",
        "| correct letter:",
        dict(Counter(row["letter_correct"] for row in rows_out if row["language"] == "en")),
    )


def main(argv: Sequence[str] | None = None, download: Callable[[str], bytes] = pinned.download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", type=Path, required=True, help="folder of the two datasets' files")
    parser.add_argument("--translation", type=Path, required=True, help="the manifest the translation step wrote")
    parser.add_argument("--out", type=Path, required=True, help="folder for squad_in.jsonl and belebele_in.jsonl")
    args = parser.parse_args(argv)
    fetch(args.sources, download)
    fixed = json.loads(args.translation.read_text(encoding="utf-8"))["languages"]
    line = LETTER_INSTRUCTION
    closing = {"en": line} | {code: fixed[code]["fixed_strings"][line] for code in ("hi", "ar", "ru")}
    print("closing lines:", closing)
    args.out.mkdir(parents=True, exist_ok=True)
    squad(args.sources / SQUAD_FOLDER / SQUAD_FILE, args.out)
    belebele(args.sources / BELEBELE_FOLDER / "data", closing, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
