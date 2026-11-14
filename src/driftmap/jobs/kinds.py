"""Job kinds and what each needs on the Hub before it may run."""

import math
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
    "repair": _kind("repair", ("model", "merged", "eval", "responses"), "alpha"),
    "steer": _kind("steer", ("merged", "eval"), BATCH),
    "quant": _kind("quant", ("repaired",), None, gpu=False),
    "quant_eval": _kind("quant_eval", ("gguf", "eval"), BATCH),
}


@dataclass(frozen=True)
class State:
    """What exists on the Hub for the step about to be submitted."""

    languages: tuple[str, ...]
    # the generate target: baseline, drifted or repaired
    checkpoint: str = "baseline"
    # the alpha the repair run is asked to push; None when it screens
    apply: float | None = None
    merged_sha256: str | None = None
    # languages with benign label rows of the drifted checkpoint, the screen's chrF reference
    benign_labels: frozenset[str] = frozenset()
    repaired_revision: str | None = None
    gguf_record: bool = False
    recover_status: str | None = None
    # the alpha a decided screen names
    chosen: float | None = None
    steer_comparison: bool = False


def _same_alpha(asked: float, chosen: float) -> bool:
    # as the apply job compares them
    return math.isclose(asked, chosen, rel_tol=0, abs_tol=1e-9)


def prerequisites(kind: str, state: State) -> list[str]:
    """Inputs the kind still lacks, empty when it may be submitted; KeyError for an unknown kind."""
    KINDS[kind]
    missing = []
    if kind == "generate" and state.checkpoint == "drifted" and not state.merged_sha256:
        missing.append("drift record with merged_sha256")
    if kind == "repair":
        if not state.merged_sha256:
            missing.append("drift record with merged_sha256")
        unlabelled = [code for code in state.languages if code not in state.benign_labels]
        if unlabelled or not state.languages:
            missing.append(f"drifted benign labels for {', '.join(unlabelled) or 'every language'}")
        if state.apply is not None and (state.recover_status != "chosen" or state.chosen is None):
            missing.append("alpha screen with recover_status chosen; decide it with `driftmap repair choose`")
        elif state.apply is not None and not _same_alpha(state.apply, state.chosen):
            missing.append(f"--apply {state.chosen:g}: the screen chose that alpha, not {state.apply:g}")
    if kind == "quant" and not state.repaired_revision:
        missing.append("repaired checkpoint revision")
    if kind == "quant_eval" and not state.gguf_record:
        missing.append("gguf.json")
    if kind == "steer" and state.recover_status != "failed" and not state.steer_comparison:
        missing.append("alpha screen with recover_status failed, or the comparison flag")
    return missing
