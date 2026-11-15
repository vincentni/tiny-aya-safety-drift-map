"""Free-form answers as the rows the free-form scoring reads: each answer joined with its prompt row.

    python -I -m driftmap.attribution.free_form_rows --prompts PROMPTS --out OUT base=BASE.jsonl tuned=TUNED.jsonl

``driftmap table choice`` writes the free-form answers of each arm to a file of their own, beside the table.
``--prompts`` is a folder of the multiple-choice prompt files, ``<language>.jsonl``: ``choice/prompts/benign/`` of
the evaluation dataset at the commit the table names for the arm. An answer is joined with the prompt row of its
language and prompt id. The written row is that prompt row, which holds the context, the question, the condition,
the options and the correct letter, with the ``response`` added; the rows keep the order of the answers.
``NAME=FILE`` writes ``NAME.jsonl``, a bare ``FILE`` keeps its name.

An answer with no prompt row, with one that is not free-form, or whose prompt text is not the row's stops the run
before anything is written: the prompt files are then not the ones the answers were given to.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# what the scoring reads of a row beside its ids and the response
CARRIED = ("context", "question", "condition", "options", "letter_correct")
FREE_FORM = "free_form"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def prompt_rows(folder: Path) -> dict[tuple[str, str], dict]:
    """The prompt rows under ``folder`` by language and prompt id; stops on an id a file holds twice."""
    found: dict[tuple[str, str], dict] = {}
    for path in sorted(folder.glob("*.jsonl")):
        for row in read_jsonl(path):
            key = (row["language"], row["prompt_id"])
            if key in found:
                raise SystemExit(f"{path.name} holds {row['prompt_id']} twice")
            found[key] = row
    return found


def joined(answers: Sequence[dict], prompts: dict[tuple[str, str], dict], name: str) -> list[dict]:
    """Each answer as its prompt row with the response; stops on an answer the prompt rows do not fit."""
    rows = []
    for number, answer in enumerate(answers, start=1):
        language, prompt_id = answer["language"], answer["prompt_id"]
        prompt = prompts.get((language, prompt_id))
        where = f"{name} row {number}"
        if prompt is None:
            raise SystemExit(f"{where}: no prompt row {prompt_id} in {language}")
        if prompt.get("format") != FREE_FORM:
            raise SystemExit(f"{where}: the prompt row {prompt_id} is not a free-form one")
        if "prompt" in answer and answer["prompt"] != prompt["prompt"]:
            raise SystemExit(f"{where}: the prompt of {prompt_id} in {language} is not the one the answer was given to")
        missing = [key for key in CARRIED if key not in prompt]
        if missing:
            raise SystemExit(f"{where}: the prompt row {prompt_id} holds no {', '.join(missing)}")
        rows.append(prompt | {"response": answer["response"]})
    return rows


def _named(text: str) -> tuple[str, Path]:
    name, mark, path = text.rpartition("=")
    return (name, Path(path)) if mark else (Path(path).stem, Path(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", type=_named, nargs="+", metavar="[NAME=]FILE", help="free-form answers of an arm")
    parser.add_argument("--prompts", type=Path, required=True, help="folder of the prompt files, <language>.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="folder for the joined files")
    args = parser.parse_args(argv)
    names = [name for name, _ in args.files]
    if len(set(names)) != len(names):
        parser.error("two files would be written under one name; give each a NAME=")
    prompts = prompt_rows(args.prompts)
    written = {name: joined(read_jsonl(path), prompts, path.name) for name, path in args.files}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in written.items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (args.out / f"{name}.jsonl").write_text(text, encoding="utf-8")
        print(f"{len(rows)} rows -> {name}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
