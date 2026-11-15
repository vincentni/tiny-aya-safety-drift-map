"""Hidden states of a checkpoint at the last token of each prompt, for the activation probe.

    python -m driftmap.attribution.probe_extract --model MODEL_DIR --manifest EVAL/manifest.json \
        --input OUT/probe_train_in.jsonl --output OUT/acts/train --workers 6 --threads 10

``--model`` and ``--manifest`` are those of ``local_generate``: a local folder of the checkpoint and the manifest
of the evaluation dataset, against which the tokenizer's rendering is checked as in the generate job. ``--input``
holds rows of ``id`` and ``prompt``. A prompt is rendered by the generate job's own call under the served
template, so it is the text a stored answer was generated from, and tokenised as the job tokenises it: one BOS,
padding on the left. The model runs in float32 on the CPU; nothing is generated and no weight is changed.

Each process writes ``<shard>.npz`` under ``--output``: ``ids``, ``states`` (rows x layers x hidden size, float32:
the hidden state at the last position, the one whose next token is the answer), ``layers`` (hidden_states
indices: 0 is the embedding output) and ``prompt_tokens``. A shard that is already there is not computed again.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import local_generate, probe

BATCH_SIZE = 8


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="local folder of the checkpoint")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.json of the evaluation dataset")
    parser.add_argument("--input", required=True, type=Path, help="rows of id and prompt")
    parser.add_argument("--output", required=True, type=Path, help="folder for the shards of hidden states")
    parser.add_argument("--layers", type=int, nargs="+", default=list(probe.LAYERS), help="hidden_states indices")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=10, help="torch threads per process")
    parser.add_argument("--workers", type=int, default=1, help="processes; workers x threads may not exceed 64")
    parser.add_argument("--shard", help="K/N: the input rows at positions K, K+N, ... (set by --workers)")
    args = parser.parse_args(argv)
    if args.workers * args.threads > local_generate.MAX_THREADS:
        parser.error(f"{args.workers} workers x {args.threads} threads is over the thread cap")
    return args


def last_token_states(
    model,
    tokenizer,
    rendered: Sequence[str],
    layers: Sequence[int],
    batch_size: int = BATCH_SIZE,
    said: Callable[[int], None] | None = None,
):
    """The hidden states of ``layers`` at the last token of each rendered prompt (prompts x layers x hidden size,
    float32) and the prompts' lengths in tokens.

    Prompts are batched by length. Stops when a prompt does not start with exactly one BOS, when the padding is
    not on the left, and when the model does not return one hidden state per block and the embedding output.
    """
    import numpy as np
    import torch

    lengths = [len(tokenizer(text)["input_ids"]) for text in rendered]
    order = sorted(range(len(rendered)), key=lambda index: lengths[index])
    states = np.zeros((len(rendered), len(layers), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        picked = order[start : start + batch_size]
        batch = tokenizer([rendered[index] for index in picked], return_tensors="pt", padding=True)
        longest = batch["input_ids"][-1]
        if longest[0].item() != tokenizer.bos_token_id or longest[1].item() == tokenizer.bos_token_id:
            raise SystemExit("the rendered prompt does not start with exactly one BOS")
        if not bool(batch["attention_mask"][:, -1].all()):
            raise SystemExit("padding is not on the left")
        with torch.no_grad():
            hidden = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True
            ).hidden_states
        if len(hidden) != model.config.num_hidden_layers + 1:
            raise SystemExit(f"the model returned {len(hidden)} hidden states")
        for place, layer in enumerate(layers):
            states[picked, place] = hidden[layer][:, -1, :].numpy()
        if said:
            said(start + len(picked))
    return states, lengths


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.workers > 1 and not args.shard:
        command = [sys.executable, "-m", __spec__.name, *(sys.argv[1:] if argv is None else argv)]
        workers = [subprocess.Popen([*command, "--shard", f"{k}/{args.workers}"]) for k in range(args.workers)]
        return max(worker.wait() for worker in workers)
    shard, of = (int(part) for part in (args.shard or "0/1").split("/"))
    target = args.output / f"{shard}.npz"
    if target.is_file():
        print(f"[{shard}/{of}] already written", flush=True)
        return 0
    # nothing is asked of the Hub; the libraries read this at import, so they are imported after it
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if of > 1:
        local_generate.pin_to_node(shard)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    rows = local_generate.read_jsonl(args.input)[shard::of]
    tokenizer = local_generate.load_rendering(args.model, args.manifest, "keep")
    tokenizer.padding_side = "left"
    rendered = [local_generate.render_row(tokenizer, row).rendered_prompt for row in rows]
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=False)
    model.eval()
    print(f"[{shard}/{of}] {len(rows)} rows, model loaded in {time.monotonic() - started:.0f}s", flush=True)
    began = time.monotonic()

    def said(done: int) -> None:
        if done == len(rows) or (done // args.batch_size) % 5 == 1:
            print(
                f"[{shard}/{of}] {done}/{len(rows)} rows, {(time.monotonic() - began) / done:.2f}s per prompt",
                flush=True,
            )

    states, lengths = last_token_states(model, tokenizer, rendered, args.layers, args.batch_size, said)
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.output / f"{shard}.tmp.npz"
    held = {"ids": np.array([row["id"] for row in rows]), "states": states, "layers": np.array(args.layers)}
    np.savez(scratch, **held, prompt_tokens=np.array(lengths))
    os.replace(scratch, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
