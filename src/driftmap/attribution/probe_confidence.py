"""A checkpoint's confidence in answers it gave: one teacher-forced pass over each prompt and its stored answer.

    python -m driftmap.attribution.probe_confidence --model MODEL_DIR --manifest EVAL/manifest.json \
        --prompts OUT/squad_dev2_in.jsonl --answers OUT/squad_dev2_base.jsonl --output OUT/confidence.jsonl \
        --workers 6 --threads 10

``--model`` and ``--manifest`` are those of ``local_generate``. For each prompt row the rendered prompt is followed
by the tokens of the stored answer text, as the tokenizer reads that text, without an end-of-sequence token; the
pass is a plain forward call in float32 on the CPU, padded on the left as in generation. The position before each
answer token predicts it.

Output rows, appended batch by batch (a rerun skips ids already written), hold no text: ``id``,
``answer_tokens``, ``first_token_probability`` (the probability of the answer's first token at the last prompt
position), ``mean_log_probability`` (the mean log-probability of the answer's tokens) and ``greedy_tokens`` (how
many of the answer's tokens are the most likely token of the pass). An answer without text has no score.
``probe_checks confidence`` reads the file.
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution import local_generate, probe_squad

BATCH = 8


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="local folder of the checkpoint")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.json of the evaluation dataset")
    parser.add_argument("--prompts", required=True, type=Path, help="rows of id and prompt")
    parser.add_argument("--answers", required=True, type=Path, help="the checkpoint's stored answers to them")
    parser.add_argument("--output", required=True, type=Path, help="the confidence rows, JSON lines")
    parser.add_argument("--threads", type=int, default=10, help="torch threads per process")
    parser.add_argument("--workers", type=int, default=1, help="processes; workers x threads may not exceed 64")
    parser.add_argument("--shard", help="K/N: the prompt rows at positions K, K+N, ... (set by --workers)")
    parser.add_argument("--limit", type=int, help="only the first rows of the prompts")
    args = parser.parse_args(argv)
    if args.workers * args.threads > local_generate.MAX_THREADS:
        parser.error(f"{args.workers} workers x {args.threads} threads is over the thread cap")
    return args


def scored(logits, width: int, answer: Sequence[int]) -> dict:
    """The confidence of one row of a batch: ``logits`` are its positions' scores over the vocabulary, the answer
    sits in the last ``len(answer)`` of its ``width`` positions."""
    import torch

    if not answer:
        return {"first_token_probability": None, "mean_log_probability": None, "greedy_tokens": 0}
    first = width - len(answer) - 1
    spread = torch.log_softmax(logits[first : width - 1].double(), dim=-1)
    target = torch.tensor(list(answer))
    taken = spread[torch.arange(len(answer)), target]
    return {
        "first_token_probability": float(taken[0].exp()),
        "mean_log_probability": float(taken.mean()),
        "greedy_tokens": int((spread.argmax(dim=-1) == target).sum()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.workers > 1 and not args.shard:
        command = [sys.executable, "-m", __spec__.name, *(sys.argv[1:] if argv is None else argv)]
        workers = [subprocess.Popen([*command, "--shard", f"{k}/{args.workers}"]) for k in range(args.workers)]
        return max(worker.wait() for worker in workers)
    shard, of = (int(part) for part in (args.shard or "0/1").split("/"))
    tag = f"[{shard}/{of}] "
    # nothing is asked of the Hub; the libraries read this at import, so they are imported after it
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if of > 1:
        local_generate.pin_to_node(shard)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    said = {name: row["response"] for name, row in probe_squad.answers_of(args.answers).items()}
    rows = local_generate.read_jsonl(args.prompts)[: args.limit][shard::of]
    done = {row["id"] for row in local_generate.read_jsonl(args.output)} if args.output.is_file() else set()
    rows = [row for row in rows if row["id"] not in done]
    lacking = [row["id"] for row in rows if row["id"] not in said]
    if lacking:
        raise SystemExit(f"{len(lacking)} rows of {args.prompts.name} have no answer in {args.answers.name}")
    print(f"{tag}{len(rows)} rows to score", flush=True)
    if not rows:
        return 0
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    tokenizer = local_generate.load_rendering(args.model, args.manifest, "keep")
    asked = [
        tokenizer(local_generate.render_row(tokenizer, {"id": row["id"], "prompt": row["prompt"]}).rendered_prompt)[
            "input_ids"
        ]
        for row in rows
    ]
    answered = [tokenizer(said[row["id"]] or "", add_special_tokens=False)["input_ids"] for row in rows]
    if any(ids_[0] != tokenizer.bos_token_id or ids_[1] == tokenizer.bos_token_id for ids_ in asked):
        raise SystemExit("a rendered prompt does not start with exactly one BOS")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, trust_remote_code=False)
    model.eval()
    order = sorted(range(len(rows)), key=lambda index: len(asked[index]) + len(answered[index]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = os.open(args.output, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    for start in range(0, len(order), BATCH):
        picked = order[start : start + BATCH]
        width = max(len(asked[index]) + len(answered[index]) for index in picked)
        ids_ = torch.full((len(picked), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(picked), width), dtype=torch.long)
        for place, index in enumerate(picked):
            whole = asked[index] + answered[index]
            # padding on the left, as in generation
            ids_[place, width - len(whole) :] = torch.tensor(whole)
            mask[place, width - len(whole) :] = 1
        with torch.no_grad():
            logits = model(input_ids=ids_, attention_mask=mask).logits
        lines = []
        for place, index in enumerate(picked):
            answer = answered[index]
            written = {"id": rows[index]["id"], "answer_tokens": len(answer)} | scored(logits[place], width, answer)
            lines.append(json.dumps(written) + "\n")
        os.write(out, "".join(lines).encode())
        finished = start + len(picked)
        if (start // BATCH) % 5 == 4 or finished == len(order):
            print(f"{tag}{finished}/{len(order)} rows", flush=True)
    os.close(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
