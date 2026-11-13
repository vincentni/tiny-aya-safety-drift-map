"""The manifest of a built evaluation set and the card rendered from it."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from driftmap.core.records import Manifest, Ref, dump
from driftmap.core.schemas import SCHEMA_MANIFEST
from driftmap.core.settings import Settings
from driftmap.data.sampler import SEED_DERIVATION


def check_licences(sources: Sequence[Ref], licences: Mapping[str, str]) -> None:
    """Raises ValidationError naming each dataset source that has no licence."""
    missing = [
        InitErrorDetails(
            type=PydanticCustomError("missing_licence", "no licence recorded for {repo}", {"repo": ref.repo}),
            loc=("licences", ref.repo),
            input=licences.get(ref.repo),
        )
        for ref in sources
        if ref.kind == "dataset" and not licences.get(ref.repo)
    ]
    if missing:
        raise ValidationError.from_exception_data(Manifest.__name__, missing)


def write(path: str | Path, settings: Settings, **fields: Any) -> Manifest:
    """Validate and write the manifest; raises ValidationError on a missing field, pin or licence."""
    manifest = Manifest(
        schema=SCHEMA_MANIFEST,
        seed=settings.seed,
        seed_derivation=SEED_DERIVATION,
        n_harmful=settings.n_harmful,
        n_heldout=settings.n_heldout,
        **fields,
    )
    check_licences(manifest.sources, manifest.licences)
    dump(manifest, Path(path))
    return manifest


def card(manifest: Manifest) -> str:
    """Markdown summary of the manifest."""
    sources = [
        f"| {ref.repo}@{ref.revision} | {ref.kind} | {manifest.licences.get(ref.repo, 'see model card')} |"
        for ref in manifest.sources
    ]
    languages = [
        f"| {language} | {counts.harmful}{' (undersampled)' if counts.undersampled else ''} "
        f"| {counts.benign_heldout} | {counts.benign_train} |"
        for language, counts in manifest.per_language.items()
    ]
    longest = ", ".join(f"{name} {tokens}" for name, tokens in manifest.max_rendered_tokens.items())
    lines = [
        "# Six-language safety evaluation set",
        "",
        "Harmful prompts sampled per language, benign held-out and fine-tuning rows, and the safe XSTest prompts.",
        "The harmful prompts come from a red-teaming dataset and are meant for safety evaluation only.",
        "",
        "| source | kind | licence |",
        "|---|---|---|",
        *sources,
        "",
        "| language | harmful | benign held-out | benign train |",
        "|---|---|---|---|",
        *languages,
        "",
        f"- seed {manifest.seed}, one random stream per language: `{manifest.seed_derivation}`",
        f"- training rows dropped as copies of an evaluation prompt: {manifest.dropped_by_dedup}",
        f"- rows dropped for special-token text: {manifest.dropped_special_token_text}",
        f"- rows dropped above the rendered length limit: {manifest.dropped_over_length}",
        f"- longest rendered prompt in tokens: {longest}",
        f"- rendered prefix sha256: `{manifest.rendered_prefix_sha256}`",
    ]
    if manifest.note:
        lines.append(f"- {manifest.note}")
    return "\n".join(lines) + "\n"
