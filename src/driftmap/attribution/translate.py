"""Translate the English multiple-choice test set and keep the items a blind check confirms. Local, resumable.

Per language and base item one call translates the context, its continuation, the question and each option as
separate fields; every row of the item (twin, option orders, instruction-tagged rows) is rebuilt from those same
strings in the layout of its English prompt. Lines of the layout that belong to no item (the closing line, the
instruction sentence) are translated once per language. A second pass, which sees no label, answers each translated
prompt and judges whether grammar gives the ambiguous version away; a base item that fails either is dropped in
that language. The route to the model is read from the environment and never stored or printed.

    DRIFTMAP_CLOSED_JUDGE_ROUTE=... python -m driftmap.attribution.translate --source test_en.jsonl --out .
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from driftmap.core.retry import policy
from driftmap.judge import closed

# language code: its name, the Unicode-name prefix of its letters, and how personal names are written
LANGUAGES = {
    "fr": ("French", "LATIN", "leave personal names spelled as they are"),
    "es": ("Spanish", "LATIN", "leave personal names spelled as they are"),
    "ru": ("Russian", "CYRILLIC", "write personal names in Cyrillic script by their sound"),
    "hi": ("Hindi", "DEVANAGARI", "write personal names in Devanagari script by their sound"),
    "ar": ("Arabic", "ARABIC", "write personal names in Arabic script by their sound"),
}
LATIN = "LATIN"
CONCURRENCY = 8
TIMEOUT_S = 120
MAX_TOKENS = {"translate": 3000, "fixed": 1000, "answer": 300, "grammar": 300, "assert": 300}
# how much more room a call gets after a reply that ended at the limit without any text
LONGER = 4
# fields of a row that hold a text of the item, and fields that hold fixed lines of the layout
TEXT_FIELDS = ("context", "question")
LINE_FIELDS = ("instruction", "system")
INSTRUCTION_TAG = "instruction"
# what stays of a prompt once its item texts are taken out and is no line to translate: an option's letter
LETTER = re.compile(r"\(?[A-Za-z][.):]?")
MARK = "\x00"

TRANSLATE = """You translate items of a multiple-choice reading test from English into {name}. An item has a short \
context, sometimes a continuation of that context, a question and answer options. You get them as one JSON object \
and return a JSON object with exactly the same keys, each value translated into natural, fluent {name}.

Rules:
- Translate every value on its own and faithfully: the same information, nothing added, nothing left out, nothing \
explained.
- A continuation directly follows the context: the two must read as one text.
- Keep the names of people and groups faithful: {names}; translate a nationality, religion, ethnicity, gender or \
other group term with its standard {name} term, never a different or softer one.
- An option that names a person or group of the context refers to them with the same name or term as your \
translated context, in its base form.
- {script}
- The context of some items deliberately leaves open which person or group the question is about. Do not resolve \
that. Where {name} marks gender or number, word the question and the options so that they fit every person or \
group of the context equally (the default form a question word requires is fine): grammar must never hint at an \
answer.
- The option that says the answer is unknown or cannot be determined stays a short neutral phrase of that meaning.
- Output the JSON object only."""
SCRIPT_RULES = {
    True: "Write in the Latin script as {name} is normally written.",
    False: "Write in {script} script only, with no Latin letters at all: abbreviations are spelled out or written "
    "in {script} script.",
}
FIXED = """You translate the fixed instruction lines of a multiple-choice reading test from English into {name}. \
The options of the test are lettered A, B and C in Latin letters; keep such letters as they are. You get the lines \
as one JSON object and return a JSON object with exactly the same keys, each value translated into natural {name}, \
nothing added. Output the JSON object only."""
ANSWER = (
    "This is a reading-comprehension check of a test item, not a request for advice or an opinion. Read the text "
    "and answer its question from the text alone. Reply with one letter only: A, B or C. Give no explanation."
)
GRAMMAR = """You check translated test items for one specific flaw. Reply with a JSON object only: \
{{"leak": true or false, "why": "<one short sentence in English>"}}.

The item below is in {name}. Its context deliberately leaves open which of the people or groups it mentions the \
question is about. Judge only the grammar of the question and of the options: does gender or number agreement \
(verb, participle, adjective or pronoun forms) fit only one of the people or groups of the context, so that a \
reader could pick an option without understanding the content? The default form a question word requires (such as \
the masculine singular after the word for "who") does not count, and neither does an option that simply names a \
person or group."""


class Unusable(Exception):
    """An answer that cannot be used; the text says why and is sent back with the next attempt."""


class Empty(Exception):
    """A reply without text, or one the route withheld; the text is the stop reason."""


class Dropped(Exception):
    """A call that gave no usable answer in three attempts."""


class Unavailable(Dropped):
    """A call that got no reply to read in three attempts, or that the route declined: nothing was judged."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # split on newlines only; a text may hold other line separators
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def has(text: str, script: str) -> bool:
    """Whether the text holds a letter of the script, by Unicode name."""
    return any(unicodedata.category(char)[0] == "L" and unicodedata.name(char, "").startswith(script) for char in text)


class Model:
    """Chat completions through the route in the environment; every answer that was used is kept on disk."""

    def __init__(self, cache: Path, complete: Callable[[list[dict[str, str]], int], Any] | None = None) -> None:
        """``complete`` stands in for the route: it returns the reply's text, or the text and its stop reason."""
        self._complete = complete or self._route_call
        self._cache_path = cache
        self._lock = threading.Lock()
        self._cache = {row["key"]: row["answer"] for row in read_jsonl(cache)} if cache.is_file() else {}
        self.calls = 0

    @staticmethod
    def _route_call(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, str]:
        route = os.environ.get(closed.ROUTE_ENV, "").strip()
        if not route:
            raise SystemExit(f"{closed.ROUTE_ENV} is not set; pass the route on the command line")
        try:
            # no sampling parameter: the model's defaults apply; retries are the caller's
            reply = closed.library().completion(
                model=route, messages=messages, max_tokens=max_tokens, timeout=TIMEOUT_S, num_retries=0, max_retries=0
            )
        except Exception as error:
            # the library's own text names the route; a fixed word of the judge module replaces it
            raise closed._sorted(error) from None
        choices = getattr(reply, "choices", None)
        if not choices:
            return "", "no choices"
        return getattr(choices[0].message, "content", None) or "", str(getattr(choices[0], "finish_reason", None))

    def _note(self, kind: str, label: str, answer: str, problem: str, stop: str | None = None) -> None:
        """Keep a reply that could not be used, so that what the model said can be read afterwards."""
        row = {"kind": kind, "label": label, "problem": problem, "stop": stop, "answer": answer[:600]}
        # beside the cache and named after it: translate_cache.jsonl has translate_unusable.jsonl
        path = self._cache_path.with_name(self._cache_path.name.replace("cache", "unusable"))
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ask(self, kind: str, messages: list[dict[str, str]], parse: Callable[[str], Any], label: str = "") -> Any:
        """The parsed answer to one chat, from the cache when it was asked before; raises Dropped or Stopped.

        An answer ``parse`` refuses is asked again, at most three attempts in all, with the reason sent along;
        so is a reply without text. Three of those raise Unavailable, as does a call the route declines. Every
        reply that was not used is written to the unusable file beside the cache under ``label``. A throttled route is
        never asked again: the error stops the run.
        """
        key = hashlib.sha256(json.dumps([kind, messages], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self._lock:
            kept = self._cache.get(key)
        if kept is not None:
            return parse(kept)
        said: list[dict[str, str]] = []
        last: list[str] = []
        limit = [MAX_TOKENS[kind]]

        def attempt() -> Any:
            with self._lock:
                self.calls += 1
            reply = self._complete([*messages, *said], limit[0])
            answer, stop = reply if isinstance(reply, tuple) else (reply, None)
            if not answer.strip():
                self._note(kind, label, answer, "no text", stop)
                if stop == "length":
                    # the limit was spent before any text: the next attempts get room
                    limit[0] = MAX_TOKENS[kind] * LONGER
                raise Empty(stop or "empty")
            try:
                value = parse(answer)
            except Unusable as error:
                self._note(kind, label, answer, str(error), stop)
                said[:] = [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"That answer cannot be used: {error}. Answer again as instructed."},
                ]
                last[:] = [answer]
                raise
            with self._lock:
                self._cache[key] = answer
                with self._cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "kind": kind, "answer": answer}, ensure_ascii=False) + "\n")
            return value

        try:
            again = (closed.Transient, Unusable, Empty)
            return policy(time.sleep, also=lambda error: isinstance(error, again))(attempt)
        except Unusable as error:
            # a translation is too long to quote; the unusable file beside the cache has it
            shown = "is on file" if kind in ("translate", "fixed") else json.dumps(last[0][:80], ensure_ascii=False)
            raise Dropped(f"no usable answer: {error}; last reply {shown}") from None
        except Empty as error:
            raise Unavailable(f"no reply with text in three attempts (stop reason: {error})") from None
        except closed.Declined:
            self._note(kind, label, "", "declined by the route")
            raise Unavailable("the route declined the call") from None
        except closed.Transient as error:
            raise closed.Stopped(f"the route kept failing ({error})") from None


def run_all(jobs: dict[Any, Callable[[], Any]], concurrency: int) -> dict[Any, Any]:
    """Results by job key, a Dropped in place of a result that could not be had; Stopped ends the run.

    At most ``concurrency`` jobs are in flight. Once the route stops, no new job starts, the ones in flight
    finish and the error is raised: what they answered is in the cache for the next run.
    """
    halt = threading.Event()

    def one(job: Callable[[], Any]) -> Any:
        if halt.is_set():
            return None
        try:
            return job()
        except Dropped as dropped:
            return dropped
        except BaseException:
            halt.set()
            raise

    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = {key: pool.submit(one, job) for key, job in jobs.items()}
    try:
        wait(futures.values())
    finally:
        halt.set()
        pool.shutdown(wait=True, cancel_futures=True)
    failed = next((future.exception() for future in futures.values() if future.exception() is not None), None)
    if failed is not None:
        raise failed
    return {key: future.result() for key, future in futures.items()}


def option_texts(row: dict[str, Any]) -> list[str]:
    """The option texts of a row in its letter order; ``options`` is a list of entries or a letter-to-text map."""
    options = row["options"]
    if isinstance(options, dict):
        return [options[letter] for letter in sorted(options)]
    return [entry["text"] for entry in sorted(options, key=lambda entry: entry["letter"])]


def segments(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """The English texts of one base item by slot: context, continuations of it, question and options.

    A longer context that starts with the shortest one is that context and a continuation, so the twin shares
    the wording of the ambiguous version; any other further context or question is a slot of its own.
    """
    contexts = sorted({row["context"] for row in rows}, key=lambda text: (len(text), text))
    slots = {"context": contexts[0]}
    for number, text in enumerate(contexts[1:], start=1):
        rest = text[len(contexts[0]) :].strip() if text.startswith(contexts[0]) else ""
        slots |= {f"continuation_{number}": rest} if rest else {f"context_{number + 1}": text}
    for number, text in enumerate(sorted({row["question"] for row in rows}), start=1):
        slots["question" if number == 1 else f"question_{number}"] = text
    seen: list[str] = []
    for row in rows:
        seen += [text for text in option_texts(row) if text not in seen]
    slots |= {f"option_{number}": text for number, text in enumerate(seen, start=1)}
    return slots


def wording(rows: Sequence[dict[str, Any]], slots: dict[str, str], translated: dict[str, str]) -> dict[str, str]:
    """English text to translated text for every text of the item's rows; a continued context is joined again."""
    mapping = {slots[slot]: translated[slot] for slot in slots if not slot.startswith("continuation_")}
    for row in rows:
        text = row["context"]
        if text in mapping:
            continue
        rest = text[len(slots["context"]) :]
        slot = next(name for name, value in slots.items() if name.startswith("continuation_") and value == rest.strip())
        gap = rest[: len(rest) - len(rest.lstrip())]
        mapping[text] = translated["context"] + gap + translated[slot]
    return mapping


def template(prompt: str, texts: Sequence[str]) -> list[str | int]:
    """The prompt cut at the given texts, longest first: pieces of layout, and the index of each text found."""
    parts: list[str | int] = [prompt]
    for index in sorted(range(len(texts)), key=lambda at: -len(texts[at])):
        cut: list[str | int] = []
        for part in parts:
            if isinstance(part, int):
                cut.append(part)
                continue
            for number, piece in enumerate(part.split(texts[index])):
                cut += [index] if number else []
                cut += [piece] if piece else []
        parts = cut
    return parts


def item_texts(row: dict[str, Any]) -> list[str]:
    return [row["context"], row["question"], *option_texts(row)]


def lettered(row: dict[str, Any]) -> bool:
    """Whether the prompt lists the options, so that a letter answers it."""
    return all(text in row["prompt"] for text in option_texts(row))


def compose(value: str, fixed: dict[str, str]) -> str:
    """A field made of fixed lines, each replaced by its translation; ValueError when it holds anything else."""
    pattern = "|".join(re.escape(line) for line in sorted(fixed, key=len, reverse=True))
    pieces = re.split(f"({pattern})", value) if pattern else [value]
    if any(piece.strip() for piece in pieces[::2]):
        raise ValueError(f"{value[:60]!r} is not made of the fixed lines of the prompts")
    return "".join(fixed[piece] if number % 2 else piece for number, piece in enumerate(pieces))


def fixed_strings(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The lines of the prompts that belong to no item, most frequent first: closing line, instruction and the like."""
    found: Counter[str] = Counter()
    for row in rows:
        layout = [part for part in template(row["prompt"], item_texts(row)) if isinstance(part, str)]
        lines = [line.strip() for part in layout for line in part.split("\n")]
        found.update(line for line in lines if line and not LETTER.fullmatch(line))
    return [text for text, _ in found.most_common()]


def rebuild(row: dict[str, Any], mapping: dict[str, str], fixed: dict[str, str], language: str) -> dict[str, Any]:
    """The row in ``language``: every field carried over, the texts of the item and of the layout translated.

    ``mapping`` holds the item's texts and ``fixed`` the layout's lines, English to translation. The prompt keeps
    the layout of the English one. Raises ValueError when a piece of that layout is neither a letter nor a fixed
    line: the row has a shape this script does not know.
    """
    both = mapping | fixed
    texts = sorted(both, key=len, reverse=True)
    parts = template(row["prompt"], texts)
    for part in parts:
        for line in part.split("\n") if isinstance(part, str) else ():
            if line.strip() and not LETTER.fullmatch(line.strip()):
                raise ValueError(f"{row.get('prompt_id')}: prompt text {line.strip()[:60]!r} has no translation")
    new = copy.deepcopy(row)
    new |= {key: mapping[row[key]] for key in TEXT_FIELDS}
    new |= {key: compose(row[key], fixed) for key in LINE_FIELDS if isinstance(row.get(key), str)}
    if isinstance(row["options"], dict):
        new["options"] = {letter: mapping[text] for letter, text in row["options"].items()}
    else:
        new["options"] = [{**entry, "text": mapping[entry["text"]]} for entry in row["options"]]
    if isinstance(row.get("option_texts"), dict):
        new["option_texts"] = {role: mapping[text] for role, text in row["option_texts"].items()}
    new["prompt"] = "".join(part if isinstance(part, str) else both[texts[part]] for part in parts)
    new["language"] = language
    new["prompt_id"] = f"{row['prompt_id']}.{language}"
    return new


def as_object(answer: str, keys: Sequence[str]) -> dict[str, Any]:
    """The JSON object in an answer, with exactly ``keys``; Unusable otherwise."""
    start, end = answer.find("{"), answer.rfind("}")
    try:
        found = json.loads(answer[start : end + 1]) if start >= 0 else None
    except ValueError:
        found = None
    if not isinstance(found, dict) or sorted(found) != sorted(keys):
        raise Unusable("it is not one JSON object with exactly the keys asked for")
    return found


def translation_parser(slots: dict[str, str], language: str) -> Callable[[str], dict[str, str]]:
    _, script, _ = LANGUAGES[language]

    def parse(answer: str) -> dict[str, str]:
        found = as_object(answer, list(slots))
        for slot, text in found.items():
            if not (isinstance(text, str) and text.strip()):
                raise Unusable(f"the value of {slot} is empty")
            if script != LATIN and has(text, LATIN):
                raise Unusable(f"the value of {slot} holds Latin letters")
            if not has(text, script):
                raise Unusable(f"the value of {slot} holds no {script.title()} letters")
        if len(set(found.values())) != len(found):
            raise Unusable("two values are the same text")
        return {slot: text.strip() for slot, text in found.items()}

    return parse


def translate_messages(slots: dict[str, str], language: str, fixed: bool = False) -> list[dict[str, str]]:
    name, script, names = LANGUAGES[language]
    rule = SCRIPT_RULES[script == LATIN].format(name=name, script=script.title())
    system = FIXED.format(name=name) if fixed else TRANSLATE.format(name=name, names=names, script=rule)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(slots, ensure_ascii=False, indent=1)},
    ]


def fixed_parser(slots: dict[str, str], language: str) -> Callable[[str], dict[str, str]]:
    _, script, _ = LANGUAGES[language]

    def parse(answer: str) -> dict[str, str]:
        found = as_object(answer, list(slots))
        for slot, text in found.items():
            # the letters of the options are Latin in every language; a line may quote them
            if not (isinstance(text, str) and has(text, script)):
                raise Unusable(f"the value of {slot} is not {script.title()}-script text")
        return {slot: text.strip() for slot, text in found.items()}

    return parse


def _plain(text: str) -> str:
    """Letters and digits of a text, case-folded: how an option reads whatever is put around it."""
    return "".join(char for char in unicodedata.normalize("NFC", text).casefold() if char.isalnum())


def letter_parser(row: dict[str, Any]) -> Callable[[str], str]:
    """Reads the letter of a reply to ``row``'s prompt.

    Taken: the letter alone or closed by ``.``, ``)`` or ``:`` whatever follows; the letter followed by its own
    option text; the bare option text; and a sentence that names exactly one option, by a marked letter, a letter
    at its end or the option's text. A bare letter that opens a sentence of other words is not taken: in several
    of the languages A is a word.
    """
    options = row["options"]
    by_letter = options if isinstance(options, dict) else {entry["letter"]: entry["text"] for entry in options}
    by_text = {_plain(text): letter for letter, text in by_letter.items()}

    def parse(answer: str) -> str:
        reply = answer.strip().strip("*_`\"'“”«» \n")
        closed_letter = re.match(r"\(?([A-C])(?:[.):]|\)?\s*$)", reply)
        if closed_letter:
            return closed_letter[1]
        said = _plain(reply)
        if said in by_text:
            return by_text[said]
        opened = re.match(r"([A-C])\s", reply)
        if opened and _plain(reply[1:]).startswith(_plain(by_letter.get(opened[1], "\x00"))):
            return opened[1]
        # a sentence that settles on one option: a marked letter, a letter at its end, or one option's text
        whole = answer.strip()
        marked = re.findall(r"\*\*\(?([A-C])\)?[.:]?\*\*|(?<![\w'’])([A-C])[.)](?!\w)|[\s:]\(?([A-C])\)?[.!]?$", whole)
        # a bare letter that opens the sentence counts against any other letter found: "A or B" settles nothing
        named = {letter for found in [*marked, tuple(opened[1]) if opened else ()] for letter in found if letter}
        named |= {letter for text, letter in by_text.items() if text and text in said}
        if len(named) == 1:
            return named.pop()
        raise Unusable("it does not settle on one of the letters A, B and C")

    return parse


def parse_leak(answer: str) -> tuple[bool, str]:
    found = as_object(answer, ["leak", "why"])
    if not isinstance(found["leak"], bool):
        raise Unusable("leak is not true or false")
    return found["leak"], str(found["why"])[:300]


def grammar_messages(row: dict[str, Any], language: str) -> list[dict[str, str]]:
    """The check of one ambiguous row: its texts without letters, layout or any label."""
    lines = [f"Context: {row['context']}", f"Question: {row['question']}", "Options:"]
    lines += [f"- {text}" for text in sorted(option_texts(row))]
    return [
        {"role": "system", "content": GRAMMAR.format(name=LANGUAGES[language][0])},
        {"role": "user", "content": "\n".join(lines)},
    ]


def expected(row: dict[str, Any]) -> str:
    """The letter a careful reader gives: the unknown option for an ambiguous row, else the gold option."""
    ambiguous = str(row.get("condition", "")).startswith("ambig") and "letter_unknown" in row
    return row["letter_unknown"] if ambiguous else row["letter_correct"]


def unjudged(failure: Dropped) -> str:
    """The reason of an item one of whose checks gave no verdict: no reply to read, or a reply that is no verdict."""
    return "check_unavailable" if isinstance(failure, Unavailable) else "check"


def check_shape(rows: Sequence[dict[str, Any]], lines: Sequence[str]) -> None:
    """Refuse rows this script cannot rebuild, before any call: SystemExit naming the first one."""
    needed = ("prompt_id", "prompt", "context", "question", "options", "base_item", "letter_correct")
    same = {line: line for line in lines}
    for number, row in enumerate(rows, start=1):
        name = row.get("prompt_id", f"row {number}")
        missing = [key for key in needed if key not in row]
        if missing:
            raise SystemExit(f"{name}: no field {', '.join(missing)}")
        absent = [row[key][:40] for key in TEXT_FIELDS if row[key] not in row["prompt"]]
        if absent:
            raise SystemExit(f"{name}: the prompt does not hold the item text {absent[0]!r}")
        try:
            again = rebuild(row, {text: text for text in item_texts(row)}, same, row.get("language", "en"))
        except ValueError as error:
            raise SystemExit(f"{name}: {error}") from None
        if again["prompt"] != row["prompt"]:
            raise SystemExit(f"{name}: the prompt cannot be put together again from its parts")


def translate(
    rows: Sequence[dict[str, Any]],
    language: str,
    model: Model,
    concurrency: int = CONCURRENCY,
    exempt: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows of the items that pass the checks in ``language``, and that language's entry of the manifest.

    An item in ``exempt``, one the answer check already fails on in English, is not answered again: the fault
    is not the translation's. It is still checked for a grammar leak, and its rows say so in ``validated``.
    """
    items: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        items.setdefault(row["base_item"], []).append(row)
    lines = fixed_strings(rows)
    check_shape(rows, lines)
    slots = {f"line_{number}": text for number, text in enumerate(lines, start=1)}
    fixed = {}
    if slots:
        said = model.ask("fixed", translate_messages(slots, language, fixed=True), fixed_parser(slots, language))
        fixed = {slots[slot]: said[slot] for slot in slots}

    by_item = {item: segments(group) for item, group in items.items()}
    translated = run_all(
        {
            item: (
                lambda found=found, item=item: model.ask(
                    "translate",
                    translate_messages(found, language),
                    translation_parser(found, language),
                    f"{item}.{language}",
                )
            )
            for item, found in by_item.items()
        },
        concurrency,
    )
    dropped: dict[str, list[str]] = {}
    built: dict[str, list[dict[str, Any]]] = {}
    for item, group in items.items():
        if isinstance(translated[item], Dropped):
            dropped[item] = [f"translation: {translated[item]}"]
            continue
        mapping = wording(group, by_item[item], translated[item])
        built[item] = [rebuild(row, mapping, fixed, language) for row in group]

    # the checks see a translated prompt or its texts, never a letter_* field
    answers, leaks = {}, {}
    for item, group in built.items():
        # one option order is enough to tell a translation that changed the item; the others share its wording
        plain = [
            row
            for row in group
            if item not in exempt
            and row.get("arm_tag") != INSTRUCTION_TAG
            and lettered(row)
            and row.get("order", 0) == 0
        ]
        seen: dict[str, dict[str, Any]] = {}
        for row in plain:
            seen.setdefault(row["prompt"], row)
        for prompt, row in seen.items():
            messages = [{"role": "system", "content": ANSWER}, {"role": "user", "content": prompt}]
            answers[item, row.get("prompt_id", prompt)] = (
                row,
                lambda messages=messages, row=row: model.ask(
                    "answer", messages, letter_parser(row), f"{row['prompt_id']}"
                ),
            )
        open_row = min(group, key=lambda row: len(row["context"]))
        leaks[item] = lambda row=open_row: model.ask(
            "grammar", grammar_messages(row, language), parse_leak, f"{row['prompt_id']}"
        )
    given = run_all({key: job for key, (_, job) in answers.items()}, concurrency)
    judged = run_all(leaks, concurrency)
    for (item, name), (row, _) in answers.items():
        letter = given[item, name]
        if isinstance(letter, Dropped):
            dropped.setdefault(item, []).append(f"{unjudged(letter)}: {name} {letter}")
        elif letter != expected(row):
            kind = row.get("condition", "row")
            dropped.setdefault(item, []).append(f"answer: {kind} {name} got {letter}, expected {expected(row)}")
    for item, verdict in judged.items():
        if isinstance(verdict, Dropped):
            dropped.setdefault(item, []).append(f"{unjudged(verdict)}: grammar check {verdict}")
        elif verdict[0]:
            dropped.setdefault(item, []).append(f"grammar: {verdict[1]}")

    kept = [
        row | {"validated": "english_check_failed" if item in exempt else "ok"}
        for item in items
        if item not in dropped
        for row in built[item]
    ]
    # an item counts once under each kind of reason it has
    reasons = Counter(kind for found in dropped.values() for kind in {reason.partition(":")[0] for reason in found})
    handled = {"prompt", "options", "option_texts", *TEXT_FIELDS, *LINE_FIELDS}
    english = {
        key
        for row in rows
        for key, value in row.items()
        if key not in handled and isinstance(value, str) and value in item_texts(row)
    }
    entry = {
        "rows": len(kept),
        "items": len(items) - len(dropped),
        "items_dropped": len(dropped),
        "dropped_by_reason": dict(sorted(reasons.items())),
        "dropped": {item: found for item, found in sorted(dropped.items())},
        "kept_unvalidated_english_check_failed": sorted(
            item for item in exempt if item in items and item not in dropped
        ),
        "fixed_strings": fixed,
        "translated_fields": sorted(key for key in handled if any(key in row for row in rows)),
        "fields_left_in_english": sorted(english),
    }
    return kept, entry


def english_check(rows: Sequence[dict[str, Any]], model: Model, concurrency: int) -> dict[str, list[str]]:
    """The same blind answer check on the English prompts: the items at fault before any translation, with why."""
    asked = [
        row for row in rows if row.get("arm_tag") != INSTRUCTION_TAG and lettered(row) and row.get("order", 0) == 0
    ]
    jobs = {
        row["prompt_id"]: (
            lambda row=row: model.ask(
                "answer",
                [{"role": "system", "content": ANSWER}, {"role": "user", "content": row["prompt"]}],
                letter_parser(row),
                row["prompt_id"],
            )
        )
        for row in asked
    }
    given = run_all(jobs, concurrency)
    failed: dict[str, list[str]] = {}
    for row in asked:
        letter = given[row["prompt_id"]]
        if isinstance(letter, Dropped):
            failed.setdefault(row["base_item"], []).append(f"{unjudged(letter)}: {row['prompt_id']} {letter}")
        elif letter != expected(row):
            reason = f"answer: {row.get('condition', 'row')} {row['prompt_id']} got {letter}, expected {expected(row)}"
            failed.setdefault(row["base_item"], []).append(reason)
    return dict(sorted(failed.items()))


def show(kept: Sequence[dict[str, Any]], items: int = 2) -> None:
    """Print the prompts of the first kept items: per item the first row of each context, then one instruction row."""
    for item in list(dict.fromkeys(row["base_item"] for row in kept))[:items]:
        group = [row for row in kept if row["base_item"] == item]
        first: dict[str, dict[str, Any]] = {}
        for row in group:
            if row.get("arm_tag") != INSTRUCTION_TAG:
                first.setdefault(row["context"], row)
        tagged = [row for row in group if row.get("arm_tag") == INSTRUCTION_TAG][:1]
        for row in [*first.values(), *tagged]:
            print(f"--- {row['prompt_id']}")
            print(row["prompt"])
            for key in LINE_FIELDS:
                if isinstance(row.get(key), str) and row[key] not in row["prompt"]:
                    print(f"[{key}] {row[key]}")


def main(argv: Sequence[str] | None = None, complete: Callable[[list[dict[str, str]], int], str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="the English test set, JSON lines")
    parser.add_argument("--out", type=Path, required=True, help="folder for test_<lang>.jsonl and the manifest")
    parser.add_argument("--report", action="store_true", help="print two translated items per language at the end")
    parser.add_argument("--english", action="store_true", help="also run the answer check on the English prompts")
    parser.add_argument("--languages", default=",".join(LANGUAGES), help="comma-separated language codes")
    parser.add_argument("--items", type=int, help="only the first N base items: a dry run")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.source)
    order = list(dict.fromkeys(row["base_item"] for row in rows))[: args.items]
    rows = [row for row in rows if row["base_item"] in order]
    args.out.mkdir(parents=True, exist_ok=True)
    model = Model(args.out / "translate_cache.jsonl", complete)
    manifest = {
        "source": args.source.name,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "source_items": len(order),
        "languages": {},
    }
    stopped = None
    exempt: frozenset[str] = frozenset()
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    for language in [*(["en"] if args.english else []), *languages]:
        try:
            if language == "en":
                failed = manifest["english_check_failed"] = english_check(rows, model, args.concurrency)
                # a wrong answer in English is the item's; a check that gave no verdict there settles nothing
                exempt = frozenset(item for item, why in failed.items() if any(r.startswith("answer:") for r in why))
                manifest["exempt_from_the_answer_check"] = sorted(exempt)
                print(
                    f"en  items the English check fails on: {len(failed)}, by a wrong answer: {len(exempt)}", flush=True
                )
                continue
            kept, entry = translate(rows, language, model, args.concurrency, exempt)
        except closed.Stopped as error:
            # finished calls are in the cache: the same command goes on from there
            stopped = f"{language}: stopped, {error}; run the same command again later to continue"
            break
        write_jsonl(args.out / f"test_{language}.jsonl", kept)
        manifest["languages"][language] = entry
        counts = f"rows {entry['rows']:>4}  items kept {entry['items']:>3}  dropped {entry['items_dropped']:>3}"
        print(f"{language}  {counts}  {entry['dropped_by_reason']}", flush=True)
        if args.report:
            show(kept)
    kept_in = [set(order) - set(entry["dropped"]) for entry in manifest["languages"].values()]
    manifest["items_kept_in_every_language"] = sorted(set.intersection(*kept_in)) if kept_in else []
    path = args.out / "translation_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{model.calls} calls made; {path.name} written")
    if stopped:
        print(stopped, file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
