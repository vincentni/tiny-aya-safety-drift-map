"""A fresh sample of the official red-teaming prompts, held out of the evaluation set.

    python -m driftmap.data.fresh --source SOURCE --prompts PROMPTS --tokenizer TOKENIZER --out OUT \
        --patch-commit SHA --patch-committed DATE

``--source`` is a copy of the red-teaming dataset at the revision the settings pin, ``--prompts`` the ``prompts/``
folder of the evaluation set and ``--tokenizer`` a folder of the model's tokenizer files. The rows of the source
that are no evaluation prompt pass the data build's screen and lose every exact or near copy of an evaluation
prompt and of an earlier fresh row (``cross_dedup`` at the build's threshold). Of what is left, the same number
per language is drawn by seed, stratified as the evaluation prompts were, on a seed stream of its own. Written
under ``--out``: ``heldout/prompts/harmful/<language>.jsonl``, rows of the evaluation set's shape with the harm
categories of the source, and ``heldout/manifest.json``, which counts every step. The manifest names the text
of the safety patch by its version and hash, and with ``--patch-commit`` and ``--patch-committed`` the commit of
this repository that holds that text and its date: the sample is drawn after the text was fixed.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from driftmap.core import ids, render
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import load_settings
from driftmap.data import build
from driftmap.data.sampler import SEED_DERIVATION, Row, Screen, sample, unique

GROUP, SET = "heldout", "harmful"
PER_LANGUAGE = 100


class Drawn(NamedTuple):
    """The held-out rows per language, and what each step left on the way to them."""

    rows: dict[str, list[Row]]
    counts: dict[str, Any]


def _jsonl(path: Path) -> list[Row]:
    # split on newlines only; a prompt may hold other line separators
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(language: str) -> str:
    """The name of a language's seed stream: another one than the evaluation prompts were drawn on."""
    return f"{GROUP}:{language}"


def patch_record(commit: str | None, committed: str | None) -> dict[str, str]:
    """What the manifest says of the safety patch: its version and the hash of its text, and where given the
    commit that holds the text and that commit's date.

    DriftmapError when one of the two is given without the other, the commit is no full hash, or the date is no
    ISO 8601 date and time.
    """
    record = {"version": render.PATCH_VERSION, "text_sha256": hashlib.sha256(render.PATCH.encode()).hexdigest()}
    if commit is None and committed is None:
        return record
    if commit is None or committed is None:
        raise DriftmapError(ExitCode.USAGE, "--patch-commit and --patch-committed go together")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DriftmapError(ExitCode.USAGE, "--patch-commit takes the full hash of a commit")
    try:
        dated = "T" in committed and datetime.fromisoformat(committed).tzinfo is not None
    except ValueError:
        dated = False
    if not dated:
        raise DriftmapError(ExitCode.USAGE, "--patch-committed takes a date and time with its zone, as ISO 8601")
    return record | {"text_fixed_at_commit": commit, "committed": committed}


def copies(protected: Sequence[Row], rows: Sequence[Row]) -> int:
    """How many of ``rows`` are a protected prompt, near one, or near an earlier row of their own."""
    from driftmap.data.dedup import cross_dedup

    known = {row["prompt_id"] for row in protected}
    fresh = [row for row in rows if row["prompt_id"] not in known]
    return len(rows) - len(cross_dedup(protected, fresh).rows)


def draw(
    found: Mapping[str, Sequence[Row]],
    evaluation: Mapping[str, Sequence[Row]],
    screen: Screen,
    languages: Sequence[str],
    seed: int,
    per_language: int = PER_LANGUAGE,
) -> Drawn:
    """``per_language`` rows of each language of ``found`` that are apart from the ``evaluation`` prompts.

    DriftmapError when an evaluation prompt is not among the source rows of its language, which says the source
    is another revision than the one the evaluation set was drawn from, and when a language is left with fewer
    rows than asked for.
    """
    from driftmap.data.dedup import cross_dedup

    protected = [row for language in languages for row in evaluation[language]]
    known = {row["prompt_id"] for row in protected}
    beyond, per = {}, {}
    for language in languages:
        distinct = unique(found.get(language, []))
        kept = screen.keep(distinct)
        missing = {row["prompt_id"] for row in evaluation[language]} - {row["prompt_id"] for row in distinct}
        if missing:
            fault = f"{len(missing)} evaluation prompts of {language} are not among the source rows"
            raise DriftmapError(
                ExitCode.INTEGRITY, f"{fault}; the source is not at the revision the set was drawn from"
            )
        beyond[language] = [row for row in kept if row["prompt_id"] not in known]
        per[language] = {
            "source_rows": len(found.get(language, [])),
            "distinct": len(distinct),
            "pass_the_screen": len(kept),
            "beyond_the_evaluation_set": len(beyond[language]),
        }
    deduped = cross_dedup(protected, [row for language in languages for row in beyond[language]])
    clean = {language: [row for row in deduped.rows if row["language"] == language] for language in languages}
    rows = {}
    for language in languages:
        picked, short = sample(clean[language], per_language, seed, stream(language))
        if short:
            left = f"{language} has {len(clean[language])} rows apart from the evaluation set"
            raise DriftmapError(ExitCode.USAGE, f"{left}, fewer than the {per_language} asked for")
        rows[language] = picked
        per[language] |= {
            "apart_from_every_other_prompt": len(clean[language]),
            "drawn": len(picked),
            "drawn_by_global_or_local": dict(sorted(Counter(row["global_or_local"] for row in picked).items())),
        }
    counts = {
        "near_an_evaluation_prompt": deduped.copies,
        "near_an_earlier_fresh_row": deduped.repeats,
        "per_language": per,
    }
    return Drawn(rows, counts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="a copy of the red-teaming dataset")
    parser.add_argument("--prompts", type=Path, required=True, help="the prompts/ folder of the evaluation set")
    parser.add_argument("--tokenizer", type=Path, required=True, help="a folder of the model's tokenizer files")
    parser.add_argument("--out", type=Path, required=True, help="folder the heldout/ folder is written under")
    parser.add_argument("--config", type=Path, default=Path("driftmap.toml"), help="settings file")
    parser.add_argument("--per-language", type=int, default=PER_LANGUAGE, help="rows drawn per language")
    parser.add_argument("--patch-commit", help="the commit of this repository that holds the safety patch's text")
    parser.add_argument("--patch-committed", help="the date of that commit, as git gives it with %%cI")
    args = parser.parse_args(argv)
    from driftmap.core.tokenizer import load_tokenizer
    from driftmap.data.dedup import THRESHOLD

    settings = load_settings(args.config)
    repo, languages = settings.sources.harmful, settings.languages
    revision = settings.revisions.get(repo)
    try:
        if revision is None:
            raise DriftmapError(ExitCode.USAGE, f"the settings pin no revision of {repo}")
        patch = patch_record(args.patch_commit, args.patch_committed)
        found = build._harmful(args.source, repo, revision)
        evaluation = {language: _jsonl(args.prompts / SET / f"{language}.jsonl") for language in languages}
        screen = Screen(load_tokenizer(args.tokenizer, build.CANARIES))
        drawn = draw(found, evaluation, screen, languages, settings.seed, args.per_language)
        every = [row for language in languages for row in drawn.rows[language]]
        protected = [row for language in languages for row in evaluation[language]]
        if copies(protected, every):
            raise DriftmapError(ExitCode.INTEGRITY, "a drawn row is an evaluation prompt or near another prompt")
    except DriftmapError as error:
        print(error, file=sys.stderr)
        return int(error.exit_code)
    folder = args.out / GROUP
    written = {}
    for language in languages:
        path = folder / ids.extra_prompts(GROUP, SET, language).removeprefix(f"{GROUP}/")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drawn.rows[language])
        path.write_text(text, encoding="utf-8")
        written[ids.extra_prompts(GROUP, SET, language)] = _sha256(path)
    categories = Counter(name for row in every for name in json.loads(row["harm_category"]))
    manifest = {
        "what": "harmful prompts of the red-teaming dataset that are no evaluation prompt and near none",
        "source": {
            "dataset": repo,
            "revision": revision,
            "licence": build._licence(args.source),
            "files_sha256": {path.name: _sha256(path) for path in sorted(args.source.glob("aya_*.jsonl"))},
        },
        "evaluation_prompts": {
            "rows": len(protected),
            "files_sha256": {
                f"{SET}/{language}.jsonl": _sha256(args.prompts / SET / f"{language}.jsonl") for language in languages
            },
        },
        "seed": settings.seed,
        "seed_stream": f"{SEED_DERIVATION}, with language = {stream('<language>')}",
        "safety_patch": patch,
        "stratified_by": "global_or_local",
        "dedup_threshold": THRESHOLD,
        "rows": len(every),
        "rows_per_language": args.per_language,
        **drawn.counts,
        "rows_by_harm_category": dict(sorted(categories.items())),
        "files_sha256": written,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for language, counted in drawn.counts["per_language"].items():
        print(
            f"{language}: {counted['drawn']} drawn of {counted['apart_from_every_other_prompt']} apart "
            f"({counted['beyond_the_evaluation_set']} beyond the evaluation set)"
        )
    print(f"{len(every)} rows -> {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
