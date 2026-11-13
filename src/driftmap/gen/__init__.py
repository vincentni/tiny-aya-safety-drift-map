"""Generation backends and the rows they produce."""

from ..core.records import Decoding

MAX_NEW_TOKENS = Decoding.model_fields["max_new_tokens"].default
ERROR_CHARS = 200
