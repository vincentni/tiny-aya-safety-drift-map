import hashlib
import re

import pytest

from driftmap.core.render import (
    Rendered,
    conditioned,
    over_length,
    prefix_hash,
    render,
    strip_bos,
    strip_preamble,
    token_count,
)
from driftmap.core.schemas import DriftmapError
from tests.fakes.tokenizers import (
    BOS,
    BOS_ID,
    END_OF_TURN,
    GENERATION_PROMPT,
    SAFETY,
    START_OF_TURN,
    SYSTEM,
    USER,
    build,
    chat_template,
    served_template,
    system_turn,
)

PROMPTS = ["hello world", "bonjour world", "नमस्ते 12345", "مرحبا canary", "привет world 12345"]


def _template_ids(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)["input_ids"]


def test_rendered_prompts_share_one_prefix_hash():
    """Rendered prompts open with the system turn and share one prefix hash (TS-RND-1)."""
    tokenizer = build()
    rendered = [render(tokenizer, prompt) for prompt in PROMPTS]

    for prompt, row in zip(PROMPTS, rendered, strict=True):
        assert isinstance(row, Rendered)
        assert row.rendered_prompt.startswith(system_turn())
        assert BOS not in row.rendered_prompt
        assert row.rendered_prompt.endswith(prompt + GENERATION_PROMPT)
    expected = hashlib.sha256(system_turn().encode()).hexdigest()
    assert {row.rendered_prefix_sha256 for row in rendered} == {expected}
    assert {row.prefix_tokens for row in rendered} == {4}

    other = render(build(preamble="canary"), PROMPTS[0])
    assert other.rendered_prefix_sha256 == hashlib.sha256(system_turn("canary").encode()).hexdigest()
    assert other.rendered_prefix_sha256 != expected


def test_stripped_text_tokenizes_to_one_bos():
    """Stripped text gets one leading BOS; unstripped text gets two (TS-RND-2)."""
    tokenizer = build("B")
    prompt = "hello canary"
    template_ids = _template_ids(tokenizer, prompt)
    stripped = render(tokenizer, prompt).rendered_prompt

    ids = tokenizer(stripped, add_special_tokens=True)["input_ids"]
    assert ids[0] == BOS_ID
    assert ids.count(BOS_ID) == 1
    assert ids == template_ids
    assert tokenizer(stripped, add_special_tokens=False)["input_ids"].count(BOS_ID) == 0
    assert token_count(tokenizer, stripped) == len(template_ids)

    unstripped = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    assert unstripped == BOS + stripped
    mutant = tokenizer(unstripped, add_special_tokens=True)["input_ids"]
    assert mutant[:2] == [BOS_ID, BOS_ID]
    assert mutant.count(BOS_ID) == 2
    assert mutant != template_ids


def test_variant_without_post_processor_never_adds_bos():
    tokenizer = build("A")
    stripped = render(tokenizer, "hello canary").rendered_prompt
    assert tokenizer(stripped, add_special_tokens=True)["input_ids"].count(BOS_ID) == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{BOS}a", "a"),
        (f"{BOS}{BOS}a", f"{BOS}a"),
        (f"a{BOS}b", f"a{BOS}b"),
        ("a", "a"),
    ],
)
def test_strip_bos_removes_one_leading_bos_only(text, expected):
    assert strip_bos(text, BOS) == expected


def test_strip_bos_without_a_bos_token_is_identity():
    assert strip_bos("a", None) == "a"


def test_literal_bos_in_the_user_turn_survives_the_strip():
    tokenizer = build()
    rendered = render(tokenizer, f"{BOS} hello").rendered_prompt
    assert rendered.count(BOS) == 1
    assert tokenizer(rendered, add_special_tokens=True)["input_ids"].count(BOS_ID) == 2


def test_prefix_hash_needs_a_user_turn():
    assert prefix_hash(render(build(), "hello").rendered_prompt) == hashlib.sha256(system_turn().encode()).hexdigest()
    with pytest.raises(DriftmapError) as exc:
        prefix_hash("no turn markers")
    assert exc.value.exit_code == 12


def test_over_length_is_strictly_above_the_limit():
    tokenizer = build()
    rendered = render(tokenizer, "hello world").rendered_prompt
    count = token_count(tokenizer, rendered)
    assert count == len(_template_ids(tokenizer, "hello world")) == 13
    assert not over_length(tokenizer, rendered, limit=count)
    assert over_length(tokenizer, rendered, limit=count - 1)
    assert not over_length(tokenizer, rendered)


def test_over_length_default_limit_is_1700_tokens():
    tokenizer = build()
    fixed = token_count(tokenizer, render(tokenizer, "hello").rendered_prompt) - 1
    at_limit = render(tokenizer, " ".join(["hello"] * (1700 - fixed))).rendered_prompt
    above = render(tokenizer, " ".join(["hello"] * (1701 - fixed))).rendered_prompt
    assert token_count(tokenizer, at_limit) == 1700
    assert not over_length(tokenizer, at_limit)
    assert over_length(tokenizer, above)


def _special(text):
    return re.findall(r"<\|?[A-Z_]+\|?>", text)


def _without(ids, span):
    start = next(i for i in range(len(ids)) if ids[i : i + len(span)] == span)
    return ids[:start] + ids[start + len(span) :]


def test_stripping_takes_the_safety_paragraph_out_and_leaves_every_token():
    keep = build(template=served_template())
    strip = conditioned(build(template=served_template()), "strip")
    assert conditioned(keep, "keep") is keep and keep.chat_template == served_template()
    paragraph = keep(SAFETY, add_special_tokens=False)["input_ids"]

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        kept = keep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stripped = strip.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert kept.count(SAFETY) == 1 and stripped == kept.replace(SAFETY, "")
        assert stripped.startswith(f"{BOS}{START_OF_TURN}{SYSTEM}# System Preamble\nYour information cutoff")
        assert _special(stripped) == _special(kept)
        assert _special(kept)[:5] == [BOS, START_OF_TURN, SYSTEM, END_OF_TURN, START_OF_TURN]
        assert _template_ids(strip, prompt) == _without(_template_ids(keep, prompt), paragraph)
        one, other = render(keep, prompt), render(strip, prompt)
        assert other.rendered_prompt == one.rendered_prompt.replace(SAFETY, "")
        assert other.rendered_prefix_sha256 != one.rendered_prefix_sha256
        assert other.prefix_tokens == one.prefix_tokens - len(paragraph)
    assert render(strip, PROMPTS[0]).rendered_prefix_sha256 == render(strip, PROMPTS[1]).rendered_prefix_sha256


def test_a_system_message_is_still_appended_under_both_conditions():
    messages = [{"role": "system", "content": "canary"}, {"role": "user", "content": "hello"}]
    keep = build(template=served_template())
    strip = conditioned(build(template=served_template()), "strip")
    kept = keep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stripped = strip.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert "# Developer Preamble\ncanary" in stripped and stripped == kept.replace(SAFETY, "")


@pytest.mark.parametrize(
    "template",
    [
        chat_template(),
        served_template().replace(SAFETY, SAFETY * 2),
        served_template().replace(SAFETY, "").replace("{{ message['content'] }}", SAFETY + "{{ message['content'] }}"),
        served_template().replace("{{ message['content'] }}", SAFETY + "{{ message['content'] }}"),
        served_template().replace("{{ bos_token }}", "{{ bos_token }}" + SAFETY),
        served_template().replace(f"{START_OF_TURN}{SYSTEM}# System", f"{START_OF_TURN}{USER}# System"),
        served_template().replace(END_OF_TURN, ""),
        None,
    ],
)
def test_a_template_without_one_safety_paragraph_in_its_system_turn_is_refused(template):
    with pytest.raises(DriftmapError) as exc:
        strip_preamble(template)
    assert exc.value.exit_code == 12
    tokenizer = build()
    tokenizer.chat_template = template
    with pytest.raises(DriftmapError):
        conditioned(tokenizer, "strip")
    assert tokenizer.chat_template == template


def test_stripping_changes_only_the_paragraph_in_the_template_text():
    template = served_template()
    assert strip_preamble(template) == template.replace(SAFETY, "")
    strict = template.replace("contextual safety mode", "strict safety mode")
    assert strip_preamble(strict) == template.replace(SAFETY, "")


def test_a_template_held_by_name_is_stripped_in_its_default_entry():
    named = build(template=served_template())
    named.chat_template = {"default": served_template(), "tools": "unused"}
    plain = conditioned(build(template=served_template()), "strip")

    strip = conditioned(named, "strip")

    assert strip.chat_template == {"default": plain.chat_template, "tools": "unused"}
    assert render(strip, "hello") == render(plain, "hello")
