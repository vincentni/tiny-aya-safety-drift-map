"""Build the evaluation set from the pinned sources and push it."""

import csv
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow.parquet as pq
from huggingface_hub import DatasetCard
from huggingface_hub.utils import filter_repo_objects

from driftmap.core import ids, redact, render
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import LangCounts, Manifest, Ref
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.core.tokenizer import load_tokenizer
from driftmap.data import ISO3, curator, manifest
from driftmap.data.dedup import cross_dedup
from driftmap.data.heldout import balance, carve
from driftmap.data.sampler import Row, Screen, sample, unique

# digits and five scripts; any encoding difference refuses the tokenizer
CANARIES = ("12345 67890", "hello world", "bonjour le monde", "привет мир", "नमस्ते दुनिया", "مرحبا بالعالم")
TOKENIZER_FILES = ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "config.json"]
UPLOADS = ["prompts/*", "sft_train/*", "manifest.json", "dataset_card.md"]
XSTEST_LANGUAGE = "en"
XSTEST_SAFE = 250

log = logging.getLogger(__name__)


def _pins(settings: Settings, hub: Hub) -> dict[str, str]:
    sources = settings.sources
    lookups = {
        sources.harmful: hub.dataset_info,
        sources.benign: hub.dataset_info,
        sources.xstest: hub.dataset_info,
        sources.model: hub.model_info,
    }
    return {repo: settings.revisions.get(repo) or info(repo).sha for repo, info in lookups.items()}


def _per_language(total: int, settings: Settings, name: str) -> int:
    count, rest = divmod(total, len(settings.languages))
    if rest:
        raise DriftmapError(
            ExitCode.USAGE, f"{name} = {total} does not divide over {len(settings.languages)} languages"
        )
    return count


def _licence(directory: Path) -> str:
    readme = directory / "README.md"
    licence = DatasetCard.load(readme).data.license if readme.is_file() else None
    return ", ".join(licence) if isinstance(licence, list) else licence or ""


def _row(repo: str, revision: str, language: str, kind: str, prompt: str, **extra: Any) -> Row:
    return {
        "prompt_id": ids.prompt_id(repo, revision, prompt),
        "language": language,
        "set": kind,
        "prompt": prompt,
        **extra,
    }


def _harmful(directory: Path, repo: str, revision: str) -> dict[str, list[Row]]:
    rows: dict[str, list[Row]] = {}
    for path in sorted(directory.glob("aya_*.jsonl")):
        iso3 = path.stem.removeprefix("aya_")
        language = ISO3.get(iso3, iso3)
        # split on newlines only; a prompt may hold other line separators
        with path.open(encoding="utf-8", newline="\n") as lines:
            rows[language] = [
                _row(
                    repo,
                    revision,
                    language,
                    "harmful",
                    raw["prompt"],
                    **{key: raw[key] for key in ("global_or_local", "harm_category")},
                )
                for raw in map(json.loads, filter(str.strip, lines))
            ]
    return rows


def _benign(directory: Path, repo: str, revision: str, part: str, languages: list[str]) -> dict[str, list[Row]]:
    codes = {iso3: language for iso3, language in ISO3.items() if language in languages}
    rows: dict[str, list[Row]] = {language: [] for language in languages}
    for path in sorted(directory.glob(f"data/{part}-*.parquet")):
        table = pq.read_table(
            path, columns=["inputs", "targets", "language_code"], filters=[("language_code", "in", list(codes))]
        )
        for raw in table.to_pylist():
            language = codes[raw["language_code"]]
            prompt, target = raw["inputs"] or "", raw["targets"] or ""
            rows[language].append(_row(repo, revision, language, "benign", prompt, target=target))
    return rows


def _xstest(directory: Path, repo: str, revision: str) -> list[Row]:
    with (directory / "xstest_prompts.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [
            _row(repo, revision, XSTEST_LANGUAGE, "xstest", raw["prompt"], type=raw["type"])
            for raw in csv.DictReader(handle)
            if raw["label"] == "safe"
        ]


class Built(NamedTuple):
    """A build's manifest and, after a real run, the commit its upload returned and whether it added one."""

    manifest: Manifest
    commit: str | None = None
    uploaded: bool = False


def build(settings: Settings, hub: Hub, out_dir: str | Path, *, dry_run: bool, revisions_path: str | Path) -> Built:
    """Write the evaluation set under ``out_dir``; unless ``dry_run``, pin the source revisions and upload it."""
    out_dir = Path(out_dir)
    sources, languages, seed = settings.sources, settings.languages, settings.seed
    n_harmful = _per_language(settings.n_harmful, settings, "n_harmful")
    n_heldout = _per_language(settings.n_heldout, settings, "n_heldout")
    pins = _pins(settings, hub)
    on_file = _on_file(Path(revisions_path))
    moved = sorted(repo for repo in pins if on_file.get(repo, pins[repo]) != pins[repo])
    if moved:
        raise DriftmapError(
            ExitCode.USAGE,
            f"{revisions_path} pins {', '.join(moved)} at another revision than the loaded settings; load them again",
        )
    for stale in ("prompts", "sft_train", "curatorkit"):
        shutil.rmtree(out_dir / stale, ignore_errors=True)

    with tempfile.TemporaryDirectory(prefix="driftmap-data-") as scratch:
        work = Path(scratch)

        def fetch(repo: str, patterns: list[str], repo_type: str | None = "dataset") -> Path:
            target = work / repo.replace("/", "--")
            return Path(
                hub.snapshot_download(RepoRef(repo, pins[repo]), target, repo_type=repo_type, allow_patterns=patterns)
            )

        tokenizer = load_tokenizer(fetch(sources.model, TOKENIZER_FILES, None), CANARIES)
        harmful_dir = fetch(sources.harmful, ["*.jsonl", "README.md"])
        benign_dir = fetch(sources.benign, ["data/*.parquet", "README.md"])
        xstest_dir = fetch(sources.xstest, ["*.csv", "README.md"])
        licences = {
            sources.harmful: _licence(harmful_dir),
            sources.benign: _licence(benign_dir),
            sources.xstest: _licence(xstest_dir),
        }
        screen = Screen(tokenizer)

        found = _harmful(harmful_dir, sources.harmful, pins[sources.harmful])
        skipped = {language: len(rows) for language, rows in found.items() if language not in languages}
        repeated, harmful, undersampled = 0, {}, {}
        for language in languages:
            distinct = unique(found.get(language, []))
            repeated += len(found.get(language, [])) - len(distinct)
            harmful[language], undersampled[language] = sample(screen.keep(distinct), n_harmful, seed, language)

        pools = {
            part: {
                language: screen.keep(rows, fields=("prompt", "target"))
                for language, rows in _benign(benign_dir, sources.benign, pins[sources.benign], part, languages).items()
            }
            for part in ("train", "test")
        }
        for name, rows in (("harmful", harmful), ("benign", pools["train"])):
            empty = [language for language in languages if not rows[language]]
            if empty:
                raise DriftmapError(ExitCode.USAGE, f"no {name} rows for {', '.join(empty)}")
        benign = carve(pools["train"], pools["test"], n_heldout, seed)
        safe = _xstest(xstest_dir, sources.xstest, pins[sources.xstest])
        xstest = screen.keep(unique(safe))
        if not xstest:
            raise DriftmapError(ExitCode.USAGE, f"no safe XSTest prompt is left of {len(safe)} in {sources.xstest}")

        protected = [row for rows in (*harmful.values(), *benign.heldout.values()) for row in rows]
        deduped = cross_dedup(protected, [row for rows in benign.pools.values() for row in rows])
        if not deduped.rows:
            raise DriftmapError(ExitCode.USAGE, f"no training rows left for {', '.join(languages)}")
        config = curator.render_yaml(settings, curator.stage(deduped.rows, work / "staged"), pins[sources.benign])
        result = curator.run_pipeline(config, out_dir / "curatorkit")

    cleaned: dict[str, list] = {language: [] for language in languages}
    for item in result.passed:
        cleaned[item.metadata["language"]].append(item)
    empty = [language for language in languages if not cleaned[language]]
    if empty:
        raise DriftmapError(ExitCode.USAGE, f"no training rows left for {', '.join(empty)}")
    # drawn last, so every language ends with the same count
    drawn = balance(cleaned, seed)
    trained = curator.write_sft([item for items in drawn.values() for item in items], out_dir, languages)
    for language in languages:
        curator.write_jsonl(out_dir / "prompts" / "harmful" / f"{language}.jsonl", harmful[language])
        curator.write_jsonl(out_dir / "prompts" / "benign_heldout" / f"{language}.jsonl", benign.heldout[language])
    curator.write_jsonl(out_dir / "prompts" / "xstest" / f"{XSTEST_LANGUAGE}.jsonl", xstest)
    prefix = render.render(tokenizer, "hello")
    recorded = {
        "rendered_prefix_sha256": prefix.rendered_prefix_sha256,
        "prefix_tokens": prefix.prefix_tokens,
        "tokenizer": f"{sources.model}@{pins[sources.model]}",
    }
    (out_dir / "prompts" / "rendered_prefix.json").write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    counted = f"xstest rows: {len(xstest)} of {len(safe)} safe source rows"
    if len(xstest) != XSTEST_SAFE:
        counted += f", expected {XSTEST_SAFE}"
        log.warning("%s", counted)
    origins = {name: sorted(lang for lang in languages if benign.origin[lang] == name) for name in ("test", "train")}
    notes = [
        "skipped languages: "
        + (", ".join(f"{language} {count}" for language, count in sorted(skipped.items())) or "none"),
        f"repeated harmful prompts removed before sampling: {repeated}",
        f"blank rows removed: {screen.dropped_blank}",
        f"rows quoting a home path removed: {screen.dropped_home_path}",
        f"training rows repeating an earlier training prompt removed: {deduped.repeats}",
        f"training rows rejected by the pipeline gate: {len(result.rejected)}",
        f"training rows removed by the pipeline dedup: {len(deduped.rows) - len(result.passed) - len(result.rejected)}",
        counted,
        f"held-out rows from the test split: {', '.join(origins['test']) or 'none'}",
        f"held-out rows carved from train: {', '.join(origins['train']) or 'none'}",
    ]
    written = manifest.write(
        out_dir / "manifest.json",
        settings,
        per_language={
            language: LangCounts(
                harmful=len(harmful[language]),
                undersampled=undersampled[language],
                benign_train=trained[language],
                benign_heldout=len(benign.heldout[language]),
            )
            for language in languages
        },
        sources=[
            *(Ref(kind="dataset", repo=repo, revision=pins[repo]) for repo in licences),
            Ref(kind="model", repo=sources.model, revision=pins[sources.model]),
        ],
        licences=licences,
        dropped_by_dedup=benign.duplicates + deduped.copies,
        dropped_over_length=screen.dropped_over_length,
        dropped_special_token_text=screen.dropped_special_token_text,
        max_rendered_tokens={
            "harmful": screen.max_tokens([row for rows in harmful.values() for row in rows]),
            "benign_heldout": screen.max_tokens([row for rows in benign.heldout.values() for row in rows]),
            "xstest": screen.max_tokens(xstest),
        },
        rendered_prefix_sha256=prefix.rendered_prefix_sha256,
        note="; ".join(notes),
    )
    (out_dir / "dataset_card.md").write_text(manifest.card(written), encoding="utf-8")

    relative = [path.relative_to(out_dir).as_posix() for path in sorted(out_dir.rglob("*")) if path.is_file()]
    bound = [out_dir / path for path in filter_repo_objects(relative, allow_patterns=UPLOADS)]
    findings = redact.scan(bound, data_root=out_dir)
    if findings:
        raise redact.RedactionFinding(findings)
    if dry_run:
        return Built(written)
    # pinned first: a rebuild after a failed upload reads the same revisions
    _pin(Path(revisions_path), {**on_file, **pins})
    target = f"{settings.owner}/{settings.repos.eval}"
    # visibility is the publish step's call
    hub.create_repo(target, repo_type="dataset", private=True)
    head = hub.dataset_info(target).sha
    commit = hub.upload_folder(
        target,
        out_dir,
        repo_type="dataset",
        allow_patterns=UPLOADS,
        commit_message="add evaluation and training rows",
    ).oid
    # unchanged rows make no commit: the upload answers with the head
    return Built(written, commit, commit != head)


def _pin(path: Path, pins: dict[str, str]) -> None:
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        scratch.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _on_file(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
