"""CircuitKIT's reading of the hidden states the SQuAD probe is scored on, and a layer sweep with its linear probe.

    python -I circuitkit_check.py states --model MODEL_DIR --manifest EVAL/manifest.json \
        --input OUT/squad_dev_in.jsonl --stored ACTS/squad_dev --probe OUT/probe_squad \
        --every 5 --workers 2 --threads 10 --out CHECK
    python -I circuitkit_check.py sweep --train OUT/squad_train_in.jsonl --answers OUT/squad_train_base.jsonl \
        --stored ACTS/squad_train --fit OUT/probe_squad_fit.json --out CHECK
    python -I circuitkit_check.py report --out CHECK --results RESULTS

The check runs in an environment of its own (``requirements-circuitkit.txt``); ``--package`` names the ``src``
folder of this repository where the package is not installed. Nothing is asked of the Hub and no weight is changed.

``states`` loads the base model with ``circuitkit.load_model`` from a local folder, turns the per-head hook points
off, and reads the output of one block at the last token of each prompt: ``blocks.{k-1}.hook_resid_post`` is
``hidden_states[k]`` of transformers, k being the frozen probe's layer. The prompts are rendered and tokenised by
the package's own calls, one at a time and unpadded, and the forward pass stops after that block. The frozen probe
is applied to these states and to the stored ones of the run it was scored on. ``sweep`` trains CircuitKIT's
``LinearProbe`` with its ``ProbeTrainer`` on the stored states of the train sample, one probe per stored layer,
with the study's split and label; those states were read by transformers. ``report`` writes the counts as
``probe-squad-circuitkit.json`` and ``.md`` and a row file of ids, probabilities and decisions.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import types
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

SCHEMA = "driftmap.probe-squad-circuitkit/1"
NAME = "probe-squad-circuitkit"
STATES, SWEEP, ROWS = "states.json", "sweep.json", "rows.jsonl"
MAX_THREADS = 64
EPOCHS, PATIENCE, BATCH = 50, 5, 64
LIBRARIES = ("circuitkit", "transformer_lens", "torch", "transformers")


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", type=Path, help="src folder of the repository, when driftmap is not installed")
    steps = parser.add_subparsers(dest="step", required=True)
    states = steps.add_parser("states", help="read the probe's layer through CircuitKIT and compare")
    states.add_argument("--model", required=True, type=Path, help="local folder of the base model")
    states.add_argument("--manifest", required=True, type=Path, help="manifest.json of the evaluation dataset")
    states.add_argument("--input", required=True, type=Path, help="rows of id and prompt")
    states.add_argument("--stored", required=True, type=Path, help="folder of the stored hidden states")
    states.add_argument("--probe", required=True, type=Path, help="folder of the frozen probe")
    states.add_argument("--every", type=int, default=1, help="take the rows at positions 0, N, 2N, ...")
    states.add_argument("--threads", type=int, default=10, help="torch threads per process")
    states.add_argument("--workers", type=int, default=1, help="processes; workers x threads may not exceed 64")
    states.add_argument("--shard", help="K/N: the taken rows at positions K, K+N, ... (set by --workers)")
    sweep = steps.add_parser("sweep", help="CircuitKIT's linear probe on the stored states of each layer")
    sweep.add_argument("--train", required=True, type=Path, help="rows of the train sample")
    sweep.add_argument("--answers", required=True, type=Path, help="the base model's answers to them")
    sweep.add_argument("--stored", required=True, type=Path, help="folder of their stored hidden states")
    sweep.add_argument("--fit", type=Path, help="the study's record of its own fit, for its AUC per layer")
    report = steps.add_parser("report", help="write the result files")
    report.add_argument("--results", required=True, type=Path, help="folder for the result files")
    for step in (states, sweep, report):
        step.add_argument("--out", required=True, type=Path, help="working folder of the check")
    args = parser.parse_args(argv)
    if args.step == "states" and args.workers * args.threads > MAX_THREADS:
        parser.error(f"{args.workers} workers x {args.threads} threads is over the thread cap")
    return args


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in LIBRARIES}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_states(args: argparse.Namespace, shard: int, of: int) -> None:
    """Write ``states/<shard>.npz``: ids, the probe layer's state at the last prompt token, tokens and seconds."""
    target = args.out / "states" / f"{shard}.npz"
    if target.is_file():
        print(f"[{shard}/{of}] already written", flush=True)
        return
    # the libraries read these at import, so they are imported after
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import circuitkit
    import numpy as np
    import torch

    from driftmap.attribution import local_generate, probe

    torch.set_num_threads(args.threads)
    rows = local_generate.read_jsonl(args.input)[:: args.every][shard::of]
    tokenizer = local_generate.load_rendering(args.model, args.manifest, "keep")
    frozen = probe.load(args.probe)
    name = f"blocks.{frozen.layer - 1}.hook_resid_post"
    started = time.monotonic()
    model = circuitkit.load_model(str(args.model), dtype="float32", device="cpu")
    model.eval()
    # load_model turns these on for circuit discovery; the residual stream does not need them
    model.cfg.use_attn_result = model.cfg.use_split_qkv_input = model.cfg.use_hook_mlp_in = False
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = False
    loaded = time.monotonic() - started
    print(f"[{shard}/{of}] {len(rows)} rows, model loaded in {loaded:.0f}s", flush=True)
    states = np.zeros((len(rows), model.cfg.d_model), dtype=np.float32)
    lengths, seconds = [], []
    for place, row in enumerate(rows):
        rendered = local_generate.render_row(tokenizer, {"id": row["id"], "prompt": row["prompt"]}).rendered_prompt
        tokens = tokenizer(rendered, return_tensors="pt")["input_ids"]
        if tokens[0, 0].item() != tokenizer.bos_token_id or tokens[0, 1].item() == tokenizer.bos_token_id:
            raise SystemExit("the rendered prompt does not start with exactly one BOS")
        began = time.monotonic()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[name], stop_at_layer=frozen.layer)
        seconds.append(time.monotonic() - began)
        lengths.append(tokens.shape[1])
        states[place] = cache[name][0, -1, :].float().numpy()
        if (place + 1) % 10 == 0 or place + 1 == len(rows):
            print(f"[{shard}/{of}] {place + 1}/{len(rows)} rows, {sum(seconds) / len(seconds):.2f}s per prompt",
                  flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{shard}.tmp.npz")
    np.savez(
        scratch, ids=np.array([row["id"] for row in rows]), states=states, prompt_tokens=np.array(lengths),
        seconds=np.array(seconds), load_seconds=np.array(loaded), hook=np.array(name),
        peak_rss_gib=np.array(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
    )
    os.replace(scratch, target)


def compare(args: argparse.Namespace) -> dict:
    """The frozen probe on CircuitKIT's states and on the stored ones: ``states.json`` and the row file."""
    import numpy as np

    from driftmap.attribution import probe

    asked = [row["id"] for row in probe.read_jsonl(args.input)[:: args.every]]
    read, tokens, seconds, memory, hooks = {}, [], [], [], set()
    for path in sorted((args.out / "states").glob("[0-9]*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            read.update(zip(shard["ids"].tolist(), shard["states"], strict=True))
            tokens += shard["prompt_tokens"].tolist()
            seconds += shard["seconds"].tolist()
            memory.append(float(shard["peak_rss_gib"]))
            hooks.add(str(shard["hook"]))
    if set(read) != set(asked):
        raise SystemExit(f"{len(read)} states were read for {len(asked)} rows")
    frozen = probe.load(args.probe)
    stored, layers = probe.load_states(args.stored)
    place = layers.index(frozen.layer)
    mine = np.stack([read[row_id] for row_id in asked])
    theirs = np.stack([stored[row_id][place] for row_id in asked])
    p_mine, p_theirs = frozen.probability(mine), frozen.probability(theirs)
    fired_mine, fired_theirs = p_mine >= frozen.threshold, p_theirs >= frozen.threshold
    rows = [
        {
            "id": row_id,
            "probability_circuitkit": round(float(a), 6),
            "probability_stored": round(float(b), 6),
            "fired_circuitkit": bool(c),
            "fired_stored": bool(d),
        }
        for row_id, a, b, c, d in zip(asked, p_mine, p_theirs, fired_mine, fired_theirs, strict=True)
    ]
    probe.write_jsonl(args.out / ROWS, rows)
    record = {
        "rows": len(asked),
        "rows_of_the_sample": len(probe.read_jsonl(args.input)),
        "taken": f"positions 0, {args.every}, {2 * args.every}, ... in file order" if args.every > 1 else "all",
        "layer": frozen.layer,
        "hook": hooks.pop() if len(hooks) == 1 else sorted(hooks),
        "threshold": frozen.threshold,
        "equal_decisions": int((fired_mine == fired_theirs).sum()),
        "fired_circuitkit": int(fired_mine.sum()),
        "fired_stored": int(fired_theirs.sum()),
        "largest_absolute_state_difference": float(np.abs(mine - theirs).max()),
        "largest_absolute_stored_state": float(np.abs(theirs).max()),
        "largest_probability_difference": float(np.abs(p_mine - p_theirs).max()),
        "smallest_distance_of_a_stored_probability_from_threshold": float(np.abs(p_theirs - frozen.threshold).min()),
        "mean_prompt_tokens": round(sum(tokens) / len(tokens), 1),
        "seconds_per_prompt": round(sum(seconds) / len(seconds), 2),
        "processes": len(memory),
        "threads_per_process": args.threads,
        "peak_memory_gib_per_process": round(max(memory), 1),
        "versions": _versions(),
    }
    _write(args.out / STATES, record)
    return record


def _linear_probe():
    """CircuitKIT's ``LinearProbe`` and ``ProbeTrainer``, loaded as a module of its own."""
    import circuitkit

    # the two parent packages import the pruning and plotting tools, which this environment does not hold
    for name in ("applications", "applications.common_utils"):
        parent = types.ModuleType(f"circuitkit.{name}")
        parent.__path__ = [str(Path(circuitkit.__file__).parent.joinpath(*name.split(".")))]
        sys.modules.setdefault(parent.__name__, parent)
    from circuitkit.applications.common_utils.linear_probe import LinearProbe, ProbeTrainer

    return LinearProbe, ProbeTrainer


def sweep(args: argparse.Namespace) -> dict:
    """Per stored layer, the best validation AUROC of one CircuitKIT probe: ``sweep.json``."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from driftmap.attribution import probe, probe_squad, probe_study

    linear_probe, trainer_of = _linear_probe()
    rows = probe_squad.rows_of(args.train, args.answers)
    states, layers = probe_squad.states_of(args.stored, rows)
    probe_study.squad_split(rows)
    held = np.array([row["split"] == "dev" for row in rows])
    y = torch.tensor([float(row["label"]) for row in rows])
    x = np.stack([states[row["id"]] for row in rows])
    study = {}
    if args.fit:
        fit = json.loads(args.fit.read_text(encoding="utf-8"))
        for entry in fit.get("grid") or fit["probe"]["grid"]:
            study[entry["layer"]] = max(study.get(entry["layer"], 0.0), entry["dev_auc"])
    swept = []
    for place, layer in enumerate(layers):
        torch.manual_seed(probe.SEED)
        mean, scale = x[~held, place].mean(0), x[~held, place].std(0) + 1e-6
        standard = torch.tensor((x[:, place] - mean) / scale, dtype=torch.float32)
        fit_part = DataLoader(TensorDataset(standard[~held], y[~held]), batch_size=BATCH, shuffle=True)
        dev_part = DataLoader(TensorDataset(standard[held], y[held]), batch_size=256)
        trainer = trainer_of(linear_probe(standard.shape[1]), device="cpu")
        history = trainer.train(fit_part, dev_part, epochs=EPOCHS, patience=PATIENCE, verbose=False)
        swept.append({
            "layer": layer,
            "circuitkit_dev_auc": round(float(max(history["val_auroc"])), 3),
            "epochs": len(history["val_auroc"]),
            "study_dev_auc": study.get(layer),
        })
        print(swept[-1], flush=True)
    record = {
        "fit_rows": int((~held).sum()),
        "dev_rows": int(held.sum()),
        "label": "the base model's own error",
        "states_read_by": "transformers",
        "probe": "LinearProbe trained by ProbeTrainer at its default settings, states standardised on the fit rows",
        "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH, "seed": probe.SEED,
        "layers": swept,
        "best_layer": max(swept, key=lambda entry: entry["circuitkit_dev_auc"])["layer"],
        "versions": _versions(),
    }
    _write(args.out / SWEEP, record)
    return record


def markdown(states: dict, swept: dict | None) -> str:
    versions = states["versions"]
    lines = [
        "# SQuAD 2.0: the probe's hidden states read through CircuitKIT",
        "",
        f"The base model was loaded a second time with CircuitKIT {versions['circuitkit']} (its Cohere2 port for "
        f"TransformerLens {versions['transformer_lens']}), and the output of block {states['layer'] - 1} at the "
        f"last prompt token (`{states['hook']}`) was read for {states['rows']} of the {states['rows_of_the_sample']} "
        f"dev questions: {states['taken']}. The frozen in-domain probe was applied to these states and to the "
        f"stored ones of the run it was scored on, which transformers returned as `hidden_states[{states['layer']}]`.",
        "",
        "| questions | equal probe decision | fired, CircuitKIT | fired, stored | largest state difference "
        "| largest probability difference | smallest distance of a stored probability from the threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {states['rows']} | {states['equal_decisions']} | {states['fired_circuitkit']} | {states['fired_stored']} "
        f"| {states['largest_absolute_state_difference']:.1e} "
        f"| {states['largest_probability_difference']:.1e} "
        f"| {states['smallest_distance_of_a_stored_probability_from_threshold']:.1e} |",
        "",
        f"The largest stored state value is {states['largest_absolute_stored_state']:.1f}; the threshold is "
        f"{states['threshold']}. One prompt took {states['seconds_per_prompt']} s on the CPU "
        f"({states['mean_prompt_tokens']} tokens on average, {states['processes']} processes of "
        f"{states['threads_per_process']} threads, {states['peak_memory_gib_per_process']} GiB each).",
    ]
    if swept:
        lines += [
            "",
            "## Layer sweep with CircuitKIT's linear probe",
            "",
            f"One `LinearProbe` per stored layer, trained by CircuitKIT's `ProbeTrainer` on {swept['fit_rows']} "
            f"questions of the train sample and read on its {swept['dev_rows']} held-out questions, with the study's "
            "split and label (the base model's own error). The states of this sweep are the stored ones, read by "
            "transformers. The CircuitKIT column is the best epoch on the held-out questions; the study's column "
            "is its logistic regression at the best regularisation of its grid, on the same questions.",
            "",
            "| layer | AUC, CircuitKIT probe | AUC, the study's probe |",
            "| --- | --- | --- |",
            *(
                f"| {entry['layer']} | {entry['circuitkit_dev_auc']:.3f} | "
                f"{'-' if entry['study_dev_auc'] is None else format(entry['study_dev_auc'], '.3f')} |"
                for entry in swept["layers"]
            ),
            "",
            f"The CircuitKIT probe is best at layer {swept['best_layer']}.",
        ]
    lines += [
        "",
        "## Limits",
        "",
        f"- {states['rows']} of {states['rows_of_the_sample']} questions, taken by position, not at random.",
        "- CircuitKIT read one unpadded prompt at a time; the stored run read left-padded batches.",
        "- The check covers the probe's layer. The last entry of transformers' hidden states is taken after the "
        "final norm, so it equals the last block's output only once that norm is applied.",
        "- The per-head hook points that `circuitkit.load_model` turns on were turned off; they do not change "
        "the residual stream.",
    ]
    if swept:
        lines += [
            "- The sweep's states were read by transformers, not by CircuitKIT.",
            "- Both columns of the sweep are read on the questions their probes were chosen on, so they compare "
            "layers; neither is a score on the dev sample.",
        ]
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> None:
    states = json.loads((args.out / STATES).read_text(encoding="utf-8"))
    swept = json.loads((args.out / SWEEP).read_text(encoding="utf-8")) if (args.out / SWEEP).is_file() else None
    record = {"schema": SCHEMA, **states}
    if swept:
        record["sweep"] = {key: value for key, value in swept.items() if key != "versions"}
    args.results.mkdir(parents=True, exist_ok=True)
    _write(args.results / f"{NAME}.json", record)
    (args.results / f"{NAME}.md").write_text(markdown(states, swept), encoding="utf-8")
    (args.results / f"{NAME}-rows.jsonl").write_bytes((args.out / ROWS).read_bytes())


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    sys.dont_write_bytecode = True
    if args.package:
        sys.path.insert(0, str(args.package.resolve()))
    if args.step == "sweep":
        sweep(args)
    elif args.step == "report":
        report(args)
    elif args.shard:
        read_states(args, *(int(part) for part in args.shard.split("/")))
    else:
        command = [sys.executable, "-I", __file__, *(sys.argv[1:] if argv is None else argv)]
        workers = [subprocess.Popen([*command, "--shard", f"{k}/{args.workers}"]) for k in range(args.workers)]
        if max(worker.wait() for worker in workers):
            return 1
        record = compare(args)
        print(json.dumps({key: value for key, value in record.items() if key != "versions"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
