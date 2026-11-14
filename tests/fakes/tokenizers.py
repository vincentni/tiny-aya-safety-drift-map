"""Synthetic WordLevel tokenizers with the Tiny Aya turn markers."""

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

PAD = "<PAD>"
UNK = "<unk>"
BOS = "<BOS_TOKEN>"
END_OF_TURN = "<|END_OF_TURN_TOKEN|>"
START_OF_TURN = "<|START_OF_TURN_TOKEN|>"
USER = "<|USER_TOKEN|>"
CHATBOT = "<|CHATBOT_TOKEN|>"
START_RESPONSE = "<|START_RESPONSE|>"
SYSTEM = "<|SYSTEM_TOKEN|>"
MARKERS = [END_OF_TURN, START_OF_TURN, USER, CHATBOT, START_RESPONSE, SYSTEM]
WORDS = ["preamble", "hello", "12345", "canary", "world", "bonjour", "नमस्ते", "مرحبا", "привет"]

VOCAB = {token: i for i, token in enumerate([PAD, UNK, BOS, *MARKERS, *WORDS])}
BOS_ID = VOCAB[BOS]
GENERATION_PROMPT = f"{END_OF_TURN}{START_OF_TURN}{CHATBOT}{START_RESPONSE}"


def system_turn(preamble: str = "preamble") -> str:
    return f"{START_OF_TURN}{SYSTEM}{preamble}{END_OF_TURN}"


def chat_template(preamble: str = "preamble") -> str:
    return (
        "{{ bos_token }}" + system_turn(preamble) + "{% for m in messages %}"
        f"{START_OF_TURN}{USER}" + "{{ m['content'] }}" + END_OF_TURN + "{% endfor %}"
        "{% if add_generation_prompt %}" + f"{START_OF_TURN}{CHATBOT}{START_RESPONSE}" + "{% endif %}"
    )


SAFETY = (
    "You are in contextual safety mode. You will reject requests for harmful content, but you will accept "
    "to provide information on hard topics.\n\n"
)


def served_template() -> str:
    """A template built like the served one: a literal system turn that opens with a safety paragraph."""
    return (
        "{{ bos_token }}{% set ns = namespace(system_prompt=false, expect_user=true) %}"
        "{% for message in messages %}{% if message['role']|lower == 'system' %}"
        "{% set ns.system_prompt = message['content'] %}{% break %}{% endif %}{% endfor %}"
        f"{START_OF_TURN}{SYSTEM}# System Preamble\n{SAFETY}Your information cutoff date is June 2024.\n\n"
        "# Default Preamble\nThe following instructions are your defaults.\n- Your name is Aya.\n"
        "- Prefer the active voice."
        '{% if ns.system_prompt and ns.system_prompt != "" %}\n\n# Developer Preamble\n'
        "{{ ns.system_prompt }}{% endif %}"
        f"{END_OF_TURN}"
        "{% for message in messages %}{% set role = message['role']|lower %}"
        "{% if role == 'system' and ns.system_prompt and message['content'] == ns.system_prompt %}"
        "{% continue %}{% endif %}"
        f"{START_OF_TURN}"
        "{% if role == 'user' %}" + USER + "{{ message['content'] }}"
        "{% elif role == 'assistant' or role == 'chatbot' %}"
        + f"{CHATBOT}{START_RESPONSE}"
        + "{{ message['content'] }}<|END_RESPONSE|>"
        "{% elif role == 'system' %}" + SYSTEM + "{{ message['content'] }}{% endif %}"
        f"{END_OF_TURN}"
        "{% endfor %}"
        "{% if add_generation_prompt %}" + f"{START_OF_TURN}{CHATBOT}{START_RESPONSE}" + "{% endif %}"
    )


def swapped(vocab: dict[str, int], a: str = "canary", b: str = "world") -> dict[str, int]:
    return {**vocab, a: vocab[b], b: vocab[a]}


def build(
    variant: str = "B", vocab: dict[str, int] | None = None, preamble: str = "preamble", template: str | None = None
) -> PreTrainedTokenizerFast:
    """Variant B adds BOS through a post-processor; variant A never adds one. ``template`` replaces the plain one."""
    vocab = vocab or VOCAB
    raw = Tokenizer(models.WordLevel(vocab, unk_token=UNK))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if variant == "B":
        raw.post_processor = processors.TemplateProcessing(
            single=f"{BOS} $A", pair=f"{BOS} $A $B", special_tokens=[(BOS, vocab[BOS])]
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token=BOS,
        eos_token=END_OF_TURN,
        pad_token=PAD,
        unk_token=UNK,
        additional_special_tokens=[m for m in MARKERS if m != END_OF_TURN],
    )
    tokenizer.chat_template = template or chat_template(preamble)
    return tokenizer
