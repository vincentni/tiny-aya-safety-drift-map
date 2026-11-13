"""Prompt adapter for text the chat template has already rendered."""

from auditkit.adapter import GenerationAdapter
from auditkit.model import Request
from auditkit.runspec import RunConfig
from auditkit.sample import Sample


class RawPromptAdapter(GenerationAdapter):
    """Sends each prompt verbatim; the backend must not template it again."""

    def identity(self) -> dict:
        return {**super().identity(), "apply_chat_template": False}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        requests = super().adapt(sample, config)
        for request in requests:
            request.params["apply_chat_template"] = False
        return requests
