"""Read the five labelled lines PolyGuard writes; any other output gives no labels."""

import re
from typing import NamedTuple

PARSER_VERSION = "1"

# ASCII only: Unicode case folding would take a long s for an s
_FLAGS = re.ASCII | re.IGNORECASE
_VALUE = re.compile(r"[ \t]*(yes|no)\.?(?:[ \t]+\(.*\))?[ \t]*", _FLAGS)
# line name to the label it carries; a violations line carries none but must be there
_NAMES = {
    "harmful request": "request_harm",
    "request safety violations": None,
    "response refusal": "refusal",
    "harmful response": "response_harm",
    "response safety violations": None,
}
_LINES = {name: re.compile(rf"[ \t]*{name.replace(' ', '[ \t]+')}[ \t]*:(.*)", _FLAGS) for name in _NAMES}
# every character str.splitlines() breaks at
_BREAKS = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")
# bullets, numbers, quote marks, emphasis: whatever stands before the first letter
_DECORATION = re.compile("[^A-Za-z]*")


class ParsedLabels(NamedTuple):
    request_harm: str
    refusal: str
    response_harm: str


def _repeated(raw: str) -> bool:
    """True when a label name starts more than one line, counting lines after any line break or decoration."""
    seen: set[str] = set()
    for piece in _BREAKS.split(raw):
        line = piece[_DECORATION.match(piece).end() :]
        for name, pattern in _LINES.items():
            if pattern.match(line):
                if name in seen:
                    return True
                seen.add(name)
    return False


def parse(raw: object) -> ParsedLabels | None:
    """The three yes/no labels, or None unless each of the five lines is there once and well formed; never raises."""
    if not isinstance(raw, str):
        return None
    found: dict[str, str] = {}
    for line in raw.split("\n"):
        for name, pattern in _LINES.items():
            match = pattern.fullmatch(line.removesuffix("\r"))
            if match is not None:
                found[name] = match.group(1)
    # a label given twice is not ours to choose between
    if len(found) != len(_NAMES) or _repeated(raw):
        return None
    labels = {}
    for name, field in _NAMES.items():
        if field is None:
            if not found[name].strip():
                return None
            continue
        value = _VALUE.fullmatch(found[name])
        if value is None:
            return None
        labels[field] = value.group(1).lower()
    return ParsedLabels(**labels)
