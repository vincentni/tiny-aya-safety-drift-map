"""Render a prompt through the chat template once, for every runtime."""

import hashlib
from typing import NamedTuple

from .schemas import DriftmapError, ExitCode

USER_TURN = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
MAX_RENDERED_TOKENS = 1700


class Rendered(NamedTuple):
    rendered_prompt: str
    rendered_prefix_sha256: str
    prefix_tokens: int


def strip_bos(text: str, bos_token: str | None) -> str:
    """Drop one leading BOS text; the runtime adds the BOS id itself."""
    return text.removeprefix(bos_token) if bos_token else text


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
