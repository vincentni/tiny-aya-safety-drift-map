"""Generate answers on a workstation CPU through the generate job's own rendering and backend call.

    python -m driftmap.attribution.local_generate --model MODEL_DIR --label REPO@REVISION \
        --manifest EVAL/manifest.json --input in.jsonl --output out.jsonl

For checks small enough to need no GPU job. ``--model`` is a local folder of the checkpoint, for example the target
of ``hf download REPO --revision REVISION --local-dir MODEL_DIR``; ``--label`` is written to every row as the model
it came from; ``--manifest`` is the manifest of the evaluation dataset, which the tokenizer's rendering is checked
against as in the job. Nothing is asked of the Hub, and decoding is the job's: greedy, in float32.

Input rows are plain dicts: {id, prompt, system?, opener?}; an ``opener`` is text put at the start of the assistant
turn. Output rows, appended batch by batch (a rerun skips ids already written):
{id, prompt, system?, rendered_prompt, response, generated_tokens, stop_reason, model, max_new_tokens, ...}.

``--workers N`` splits the input over N processes of ``--threads`` threads each; all append to the one output file,
so its rows are in completion order, not input order.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

DTYPE = "float32"
MAX_THREADS = 64
NODES = Path("/sys/devices/system/node")


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="local folder of the checkpoint")
    parser.add_argument("--label", required=True, help="repo@revision of the checkpoint, written to every row")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.json of the evaluation dataset")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="folder of the backend's cache; default: cache beside the output")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--preamble",
        choices=("keep", "strip", "patch"),
        default="keep",
        help="patch: the project's frozen safety patch as the system message, as the job renders it",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="the job's batch size is 16")
    parser.add_argument("--threads", type=int, default=8, help="torch threads per process")
    parser.add_argument("--workers", type=int, default=1, help="processes; workers x threads may not exceed 64")
    parser.add_argument("--shard", help="K/N: generate the input rows at positions K, K+N, ... (set by --workers)")
    parser.add_argument("--limit", type=int, help="generate at most this many of the pending rows")
    args = parser.parse_args(argv)
    if args.workers * args.threads > MAX_THREADS:
        parser.error(f"{args.workers} workers x {args.threads} threads is over the {MAX_THREADS}-thread cap")
    return args


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file; split on newlines only, as the job reads its prompt files."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def pin_to_node(index: int) -> None:
    """Keep this process and the weights it loads on one memory node, alternating by shard."""
    nodes = sorted(NODES.glob("node[0-9]*"))
    if len(nodes) < 2:
        return
    cpus = set()
    for part in (nodes[index % len(nodes)] / "cpulist").read_text().strip().split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))


def load_rendering(model_dir: Path, manifest: Path, preamble: str):
    """The checkpoint's tokenizer after the job's canary, BOS and prefix checks against the evaluation dataset's
    ``manifest``, rendering under ``preamble``."""
    from driftmap.core import render
    from driftmap.core.tokenizer import load_tokenizer
    from driftmap.scripts.drift import CANARIES
    from driftmap.scripts.generate import check_rendering

    tokenizer = load_tokenizer(model_dir, CANARIES)
    check_rendering(tokenizer, json.loads(manifest.read_text(encoding="utf-8")), preamble)
    return render.conditioned(tokenizer, preamble)


def render_row(tokenizer, row: dict, preamble: str = "keep"):
    """The job's ``Prompt`` for a row. Under ``patch`` the job's own call renders it after the frozen patch text;
    otherwise a row's ``system`` text goes in as a system message before the user turn."""
    from driftmap.core import render
    from driftmap.gen.rows import Prompt

    opener = row.get("opener")
    if opener:
        # the opener starts the assistant turn; the model continues from it
        plain = render_row(tokenizer, {key: value for key, value in row.items() if key != "opener"}, preamble)
        return plain._replace(rendered_prompt=plain.rendered_prompt + opener)
    system = row.get("system")
    if preamble == "patch":
        if system:
            raise ValueError("a row with its own system text cannot also be rendered under --preamble patch")
        return Prompt.of(tokenizer, row["id"], row["prompt"], render.system_message(preamble))
    if not system:
        return Prompt.of(tokenizer, row["id"], row["prompt"])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = render.strip_bos(text, tokenizer.bos_token)
    return Prompt(row["id"], row["prompt"], rendered, render.prefix_hash(rendered))


def generate_texts(model, prompts, config) -> list[str]:
    """Decoded answers for rendered prompts, in order, through the job's backend call; raises on a backend failure."""
    from uuid import uuid4

    from auditkit import Sample, evaluate
    from auditkit.model import AutoModel

    from driftmap.gen.adapters import RawPromptAdapter

    samples = [Sample(input=prompt.rendered_prompt, id=prompt.prompt_id) for prompt in prompts]
    # an equal fingerprint replays a stored result, skipping the model
    fresh = replace(config, extra={**config.extra, "call": uuid4().hex})
    predictions = evaluate(samples, AutoModel.resolve(model), adapter=RawPromptAdapter(), config=fresh).predictions
    if len(predictions) != len(prompts):
        raise RuntimeError(f"backend gave {len(predictions)} replies for {len(prompts)} prompts")
    return [prediction.raw_output or "" for prediction in predictions]


def fan_out(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run one process per shard and wait; the worst exit code wins."""
    command = [sys.executable, "-m", __spec__.name, *argv]
    workers = [subprocess.Popen([*command, "--shard", f"{k}/{args.workers}"]) for k in range(args.workers)]
    code = max(worker.wait() for worker in workers)
    rows = read_jsonl(args.input)
    done = {row["id"] for row in read_jsonl(args.output)} if args.output.is_file() else set()
    print(f"{args.output.name}: {len(done & {row['id'] for row in rows})}/{len(rows)} ids written, exit {code}")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.workers > 1 and not args.shard:
        return fan_out(args, sys.argv[1:] if argv is None else argv)
    # the backend's cache stays in one folder and nothing is asked of the Hub; the libraries read this at import,
    # so they are imported after it
    os.environ["XDG_CACHE_HOME"] = str((args.cache or args.output.parent / "cache").resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    shard, of = (int(part) for part in (args.shard or "0/1").split("/"))
    tag = f"[{shard}/{of}] "
    if of > 1:
        pin_to_node(shard)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)

    from driftmap.core.render import token_count
    from driftmap.gen.hf_backend import load_hf_model, run_config

    model_dir, label = args.model, args.label
    rows = read_jsonl(args.input)
    if len({row["id"] for row in rows}) != len(rows):
        print("the input repeats an id", file=sys.stderr)
        return 2
    done = {row["id"] for row in read_jsonl(args.output)} if args.output.is_file() else set()
    pending = [row for row in rows[shard::of] if row["id"] not in done][: args.limit]
    print(f"{tag}{label}: {len(rows[shard::of])} rows, {len(pending)} to generate", flush=True)
    if not pending:
        return 0

    tokenizer = load_rendering(model_dir, args.manifest, args.preamble)
    started = time.monotonic()
    model = load_hf_model(model_dir, DTYPE, args.batch_size)
    print(f"{tag}loaded in {time.monotonic() - started:.0f}s, {torch.get_num_threads()} threads, {DTYPE}", flush=True)
    config = replace(run_config(args.batch_size), max_tokens=args.max_new_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # one write per batch on an append-only descriptor, so processes sharing the file never interleave rows
    out = os.open(args.output, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    tokens = failed = 0
    spent = 0.0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        began = time.monotonic()
        prompts = [render_row(tokenizer, row, args.preamble) for row in batch]
        try:
            texts = generate_texts(model, prompts, config)
        except Exception as error:
            # nothing written for this batch, so a rerun generates it again; later batches still run
            failed += len(batch)
            print(
                f"{tag}backend error on a batch of {len(batch)}: {type(error).__name__}: {error}"[:400],
                file=sys.stderr,
                flush=True,
            )
            continue
        seconds = time.monotonic() - began
        spent += seconds
        lines = []
        for row, prompt, text in zip(batch, prompts, texts, strict=True):
            # the job's counting rule: the decoded text re-tokenised, "length" once that reaches the limit
            generated = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
            written = {"id": row["id"], "prompt": row["prompt"]}
            if row.get("system"):
                written["system"] = row["system"]
            written |= {
                "rendered_prompt": prompt.rendered_prompt,
                "response": text,
                "generated_tokens": generated,
                "stop_reason": "length" if generated >= args.max_new_tokens else "eos",
                "model": label,
                "max_new_tokens": args.max_new_tokens,
                "preamble": args.preamble,
                "prompt_tokens": token_count(tokenizer, prompt.rendered_prompt),
                "error": None if text.strip() else "empty",
                "batch_seconds": round(seconds, 1),
            }
            if row.get("opener"):
                # judged as one answer: what the assistant turn holds from its start
                written |= {"continuation": text, "response": row["opener"] + text}
            written |= {key: row[key] for key in row if key not in written}
            lines.append(json.dumps(written, ensure_ascii=False) + "\n")
            tokens += generated
        os.write(out, "".join(lines).encode())
        os.fsync(out)
        finished = min(start + args.batch_size, len(pending))
        print(
            f"{tag}{finished}/{len(pending)} rows, batch {seconds:.0f}s, "
            f"{tokens / max(spent, 1e-9):.1f} tokens/s, {spent / finished:.1f}s per prompt",
            flush=True,
        )
    os.close(out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
