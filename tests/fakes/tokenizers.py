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


def swapped(vocab: dict[str, int], a: str = "canary", b: str = "world") -> dict[str, int]:
    return {**vocab, a: vocab[b], b: vocab[a]}


def build(
    variant: str = "B", vocab: dict[str, int] | None = None, preamble: str = "preamble"
) -> PreTrainedTokenizerFast:
    """Variant B adds BOS through a post-processor; variant A never adds one."""
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
    tokenizer.chat_template = chat_template(preamble)
    return tokenizer
