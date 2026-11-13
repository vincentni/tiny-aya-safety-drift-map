# driftmap

Measures how benign fine-tuning erodes refusal behaviour in Cohere's Tiny Aya Global,
language by language, repairs it with SafeTune, and re-checks the repair after 4-bit
quantization. Written for the Lexsi hackathon 2026.

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run pytest -q -m "not smoke"

Licensed under Apache-2.0, see `LICENSE`.
