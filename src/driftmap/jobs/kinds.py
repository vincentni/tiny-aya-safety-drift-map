"""Job kinds and what each needs on the Hub before it may run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    module: str
    gpu: bool
    # field of Settings.images
    image: str
    # fields of Settings.sources and Settings.repos the job downloads
    reads: tuple[str, ...]
    resume_key: str | None


def _kind(name: str, reads: tuple[str, ...], resume_key: str | None, gpu: bool = True) -> Kind:
    return Kind(f"driftmap.scripts.{name}", gpu, "pytorch", reads, resume_key)


BATCH = "(set, language)"
KINDS: dict[str, Kind] = {
    "drift": _kind("drift", ("model", "eval"), "step"),
    "generate": _kind("generate", ("model", "merged", "repaired", "eval"), BATCH),
    "judge": _kind("judge", ("responses", "eval"), "language"),
    "repair": _kind("repair", ("model", "merged", "eval", "responses"), "alpha"),
    "steer": _kind("steer", ("merged", "eval"), BATCH),
    "quant": _kind("quant", ("repaired",), None, gpu=False),
    "quant_eval": _kind("quant_eval", ("gguf", "eval"), BATCH),
}


@dataclass(frozen=True)
class State:
    """What exists on the Hub for the step a chain is about to submit."""

    languages: tuple[str, ...]
    # the generate target: baseline, drifted or repaired
    checkpoint: str = "baseline"
    merged_sha256: str | None = None
    responses_complete: bool = False
    baseline_labels: frozenset[str] = frozenset()
    screen_responses: bool = False
    repaired_revision: str | None = None
    gguf_record: bool = False
    recover_status: str | None = None
    steer_comparison: bool = False


def prerequisites(kind: str, state: State) -> list[str]:
    """Inputs the kind still lacks, empty when it may be submitted; KeyError for an unknown kind."""
    KINDS[kind]
    missing = []
    if kind == "generate" and state.checkpoint == "drifted" and not state.merged_sha256:
        missing.append("drift record with merged_sha256")
    if kind == "judge" and not state.responses_complete:
        missing.append("complete harmful and xstest responses for the checkpoint")
    if kind == "repair":
        unlabelled = [code for code in state.languages if code not in state.baseline_labels]
        if unlabelled or not state.languages:
            missing.append(f"baseline labels for {', '.join(unlabelled) or 'every language'}")
        if not state.screen_responses:
            missing.append("drifted benign responses for the screen prompts")
    if kind == "quant" and not state.repaired_revision:
        missing.append("repaired checkpoint revision")
    if kind == "quant_eval" and not state.gguf_record:
        missing.append("gguf.json")
    if kind == "steer" and state.recover_status != "failed" and not state.steer_comparison:
        missing.append("alpha screen with recover_status failed, or the comparison flag")
    return missing
