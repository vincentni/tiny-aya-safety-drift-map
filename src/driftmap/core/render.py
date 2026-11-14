"""Render a prompt through the chat template once, for every runtime."""

import hashlib
import re
from typing import NamedTuple

from .schemas import DriftmapError, ExitCode

USER_TURN = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
SYSTEM_TURN = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
END_OF_TURN = "<|END_OF_TURN_TOKEN|>"
MAX_RENDERED_TOKENS = 1700
# the system turn's opening paragraph, up to its blank line
_SAFETY = re.compile(r"You are in \w+ safety mode\..*?\n\n", re.DOTALL)


class Rendered(NamedTuple):
    rendered_prompt: str
    rendered_prefix_sha256: str
    prefix_tokens: int


def strip_bos(text: str, bos_token: str | None) -> str:
    """Drop one leading BOS text; the runtime adds the BOS id itself."""
    return text.removeprefix(bos_token) if bos_token else text


def strip_preamble(template: object) -> str:
    """The chat template without the safety paragraph of its system turn; every token and all other text stays."""
    head, marker, rest = template.partition(SYSTEM_TURN) if isinstance(template, str) else ("", "", "")
    turn, end, tail = rest.partition(END_OF_TURN)
    stripped, found = _SAFETY.subn("", turn)
    if not (marker and end) or found != 1 or _SAFETY.search(head + tail):
        raise DriftmapError(ExitCode.INTEGRITY, "the chat template has no single safety paragraph in its system turn")
    return head + marker + stripped + end + tail


def conditioned(tokenizer, preamble: str):
    """The tokenizer rendering under ``preamble``: ``strip`` swaps its template for the stripped one, in place."""
    if preamble == "strip":
        probe = [{"role": "user", "content": "probe"}]
        # some tokenizers hold their templates by name; only the default one is rendered here
        held = tokenizer.chat_template
        named = isinstance(held, dict)
        template = strip_preamble(held.get("default") if named else held)
        served = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        tokenizer.chat_template = {**held, "default": template} if named else template
        stripped = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        # the two renderings may differ by that one paragraph and nothing else
        cut, found = _SAFETY.subn("", served)
        if found != 1 or cut != stripped:
            raise DriftmapError(ExitCode.INTEGRITY, "the stripped template renders more than the safety paragraph away")
    return tokenizer


def _prefix(rendered_prompt: str) -> str:
    prefix, marker, _ = rendered_prompt.partition(USER_TURN)
    if not marker:
        raise DriftmapError(ExitCode.INTEGRITY, "rendered prompt has no user turn")
    return prefix


def prefix_hash(rendered_prompt: str) -> str:
    """SHA-256 of the template text before the user turn."""
    return hashlib.sha256(_prefix(rendered_prompt).encode()).hexdigest()


def token_count(tokenizer, text: str) -> int:
    """Token ids the runtime evaluates for ``text``, BOS included."""
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def over_length(tokenizer, rendered_prompt: str, limit: int = MAX_RENDERED_TOKENS) -> bool:
    return token_count(tokenizer, rendered_prompt) > limit


def render(tokenizer, prompt: str) -> Rendered:
    """Apply the chat template to one user turn, without the BOS text."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    rendered_prompt = strip_bos(text, tokenizer.bos_token)
    prefix = _prefix(rendered_prompt)
    return Rendered(
        rendered_prompt,
        hashlib.sha256(prefix.encode()).hexdigest(),
        len(tokenizer(prefix, add_special_tokens=False)["input_ids"]),
    )
