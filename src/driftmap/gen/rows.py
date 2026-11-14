"""One generation row per prompt, whatever the backend did."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, NamedTuple, Self
from uuid import uuid4

from auditkit import ChrF, Sample, evaluate
from auditkit.adapter import Adapter
from auditkit.model import AutoModel, Model, Request, Result_
from auditkit.runspec import RunConfig

from ..core.errors import OutOfMemory, chain, classify_oom
from ..core.records import Decoding, GenerationRow, LabelRow
from ..core.render import render, token_count
from ..core.schemas import SCHEMA_GENERATION, SCHEMA_LABEL, DriftmapError, ExitCode
from . import ERROR_CHARS


class Prompt(NamedTuple):
    prompt_id: str
    prompt: str
    rendered_prompt: str
    rendered_prefix_sha256: str

    @classmethod
    def of(cls, tokenizer, prompt_id: str, prompt: str, system: str | None = None) -> Self:
        """Render ``prompt`` through the chat template, after ``system`` as the system message."""
        rendered = render(tokenizer, prompt, system)
        return cls(prompt_id, prompt, rendered.rendered_prompt, rendered.rendered_prefix_sha256)


class _Counted(Model):
    """Fails a batch whose reply count differs from its request count."""

    def __init__(self, inner: Model) -> None:
        self._inner = inner

    def generate(self, requests: list[Request]) -> list[Result_]:
        results = self._inner.generate(requests)
        if len(results) != len(requests):
            raise DriftmapError(ExitCode.INTEGRITY, f"backend gave {len(results)} replies for {len(requests)} prompts")
        return results


def _stop_reason(reply: Any, generated: int, limit: int) -> str | None:
    if (reply is not None and reply.stop_type == "limit") or generated >= limit:
        return "length"
    if reply is None or reply.stop_type == "eos":
        return "eos"
    return None


def run_batch(
    model: Any,
    prompts: Iterable[Prompt],
    *,
    adapter: Adapter,
    config: RunConfig,
    tokenizer,
    checkpoint: str,
    set: str,
    language: str,
    decoding: Decoding,
) -> list[GenerationRow]:
    """One row per prompt, in order; raises ``OutOfMemory``, any other backend error leaves the batch unscored."""
    # a wrapped callable keeps its side maps on the function
    source = getattr(model, "_fn", model)
    if decoding.runtime == "llama-server" and not (hasattr(source, "failed") and hasattr(source, "replies")):
        raise DriftmapError(ExitCode.USAGE, "a llama-server batch needs the callable's failed and replies side maps")
    if decoding.max_new_tokens != config.max_tokens:
        limits = f"rows record {decoding.max_new_tokens} new tokens, the run generates {config.max_tokens}"
        raise DriftmapError(ExitCode.USAGE, limits)
    prompts = list(prompts)
    samples = [Sample(input=prompt.rendered_prompt, id=prompt.prompt_id) for prompt in prompts]
    # an equal fingerprint replays a stored result, skipping the model
    fresh = replace(config, extra={**config.extra, "call": uuid4().hex})
    failure = own = None
    try:
        counted = _Counted(AutoModel.resolve(model))
        predictions = evaluate(samples, counted, adapter=adapter, config=fresh).predictions
        outputs = [prediction.raw_output or "" for prediction in predictions]
    except Exception as error:
        if classify_oom(error):
            raise OutOfMemory(str(error)[:ERROR_CHARS]) from error
        own = next((link for link in chain(error) if isinstance(link, DriftmapError)), None)
        failure = str(error)[:ERROR_CHARS] or type(error).__name__
        outputs = [""] * len(prompts)
    if own is not None:
        raise own

    failed = getattr(source, "failed", {})
    replies = getattr(source, "replies", {})
    rows = []
    for prompt, text in zip(prompts, outputs, strict=True):
        reply = replies.get(prompt.rendered_prompt)
        generated = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
        if failure or prompt.rendered_prompt in failed:
            error, stop_reason = failure or failed[prompt.rendered_prompt], "error"
        elif reply is not None and reply.truncated:
            error = stop_reason = "truncated"
        else:
            error = None if text.strip() else "empty"
            stop_reason = _stop_reason(reply, generated, decoding.max_new_tokens)
        rows.append(
            GenerationRow(
                schema=SCHEMA_GENERATION,
                checkpoint=checkpoint,
                language=language,
                set=set,
                prompt_id=prompt.prompt_id,
                prompt=prompt.prompt,
                rendered_prompt=prompt.rendered_prompt,
                rendered_prefix_sha256=prompt.rendered_prefix_sha256,
                response=text if error is None else None,
                decoding=decoding,
                status="ok" if error is None else "unscored",
                error=error,
                prompt_tokens=token_count(tokenizer, prompt.rendered_prompt),
                generated_tokens=generated,
                stop_reason=stop_reason,
            )
        )
    return rows


def benign_rows(rows: Iterable[GenerationRow], targets: Mapping[str, str]) -> list[LabelRow]:
    """Public chrF label rows for benign generations; an unscored row has no score."""
    metric = ChrF(beta=1)
    labels = []
    for row in rows:
        sample = Sample(input=row.prompt, target=targets[row.prompt_id])
        labels.append(
            LabelRow(
                schema=SCHEMA_LABEL,
                checkpoint=row.checkpoint,
                language=row.language,
                set=row.set,
                prompt_id=row.prompt_id,
                prompt=row.prompt,
                rendered_prefix_sha256=row.rendered_prefix_sha256,
                request_harm="n/a",
                refusal="n/a",
                response_harm="n/a",
                raw_output_present=row.response is not None,
                judge=None,
                chrf=metric.score(sample, row.response).value if row.status == "ok" else None,
            )
        )
    return labels
