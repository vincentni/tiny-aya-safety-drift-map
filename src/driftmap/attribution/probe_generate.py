"""Answers of a checkpoint and the hidden states of the same run, with a probe's gate served inside it on request.

    python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION \
        --manifest EVAL/manifest.json --input IN.jsonl --output OUT.jsonl --states ACTS --max-new-tokens 48 \
        --workers 6 --threads 10
    python -m driftmap.attribution.probe_generate --model MODEL_DIR --label REPO@REVISION \
        --manifest EVAL/manifest.json --input IN.jsonl --output SERVED.jsonl --gate PROBE --max-new-tokens 24

``--model``, ``--label`` and ``--manifest`` are those of ``local_generate``. One call of ``model.generate`` per
batch: greedy, float32 on the CPU, the served template, padding on the left, the answer decoded as the decoded
sequence minus the decoded prompt. ``--states`` keeps the hidden states of the prompt pass at the last prompt token
(rows x layers x hidden size) in ``<shard>.<chunk>.npz``, in the shape ``probe.load_states`` reads, so answers and
states come from one run.

``--gate`` serves the frozen probe stored in that folder inside the run. A forward hook on the block whose output
is the probe's layer reads the last prompt token during the prompt pass and applies the probe. For a row where it
fires, a logits processor forces the token of the row's ``letter_unknown``, or with ``--abstention TEXT`` the
tokens of that one sentence, and then the end of the sequence: nothing is generated freely for that row. A row
where it does not fire generates as usual.

Output rows, appended chunk by chunk (a rerun skips ids already written), hold no prompt text: ``id``,
``response``, ``generated_tokens``, ``stop_reason``, ``first_token_id``, ``prompt_tokens``, ``model``,
``max_new_tokens`` and, under ``--gate``, ``probe_probability``, ``fired``, ``forced_token_id``,
``forced_tokens`` and ``hook_minus_hidden_state``, the largest absolute difference between the hook's state and
the run's own hidden state of that layer.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution import local_generate, probe

# batches per write
CHUNK = 5
LETTERS = "ABC"


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="local folder of the checkpoint")
    parser.add_argument("--label", required=True, help="repo@revision of the checkpoint, written to every row")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.json of the evaluation dataset")
    parser.add_argument("--input", required=True, type=Path, help="rows of id and prompt")
    parser.add_argument("--output", required=True, type=Path, help="answers, JSON lines")
    parser.add_argument("--states", type=Path, help="folder for the hidden states of the prompt pass")
    parser.add_argument("--gate", type=Path, help="folder of a frozen probe to serve inside the run")
    parser.add_argument("--abstention", help="with --gate: the sentence a fired row answers, instead of a letter")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=10, help="torch threads per process")
    parser.add_argument("--workers", type=int, default=1, help="processes; workers x threads may not exceed 64")
    parser.add_argument("--shard", help="K/N: the input rows at positions K, K+N, ... (set by --workers)")
    parser.add_argument("--limit", type=int, help="only the first rows of the input")
    args = parser.parse_args(argv)
    if args.workers * args.threads > local_generate.MAX_THREADS:
        parser.error(f"{args.workers} workers x {args.threads} threads is over the thread cap")
    if args.abstention and not args.gate:
        parser.error("--abstention goes with --gate")
    return args


class Gate:
    """The served gate of one batch: what the hook read, and which rows are forced to which tokens.

    It is the forward hook of the block and a logits processor of the same ``generate`` call. ``eos`` is the
    token that ends a forced answer.
    """

    def __init__(self, frozen: probe.Probe, eos: int) -> None:
        self.frozen, self.eos = frozen, eos
        self.reset([], 0)

    def reset(self, tokens: list[list[int]], prompt_length: int) -> None:
        """Start a batch: per row the tokens to force should the probe fire, and the batch's prompt width."""
        self.tokens, self.prompt_length = tokens, prompt_length
        self.state = self.probability = self.fired = None

    def hook(self, module, inputs, output) -> None:
        if self.probability is None:
            # the first pass of a generate call is the one over the whole prompt
            hidden = output[0] if isinstance(output, tuple) else output
            self.state = hidden[:, -1, :].detach().float().numpy().copy()
            self.probability = self.frozen.probability(self.state)
            self.fired = self.probability >= self.frozen.threshold

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        for row in range(scores.shape[0]):
            if self.fired[row]:
                token = self.tokens[row][step] if step < len(self.tokens[row]) else self.eos
                kept = scores[row, token].clone()
                scores[row, :] = float("-inf")
                scores[row, token] = kept
        return scores


def forced_tokens(
    tokenizer, abstention: str | None, max_new_tokens: int
) -> tuple[dict[str, list[int]], list[int] | None]:
    """The token of each option letter, and the tokens of the abstention sentence when one is given; stops when
    a letter is not one token or the sentence does not survive tokenisation within the token limit."""
    letters = {}
    for letter in LETTERS:
        found = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(found) != 1 or tokenizer.decode(found) != letter:
            raise SystemExit(f"the letter {letter} is not one token")
        letters[letter] = found
    sentence = None
    if abstention:
        sentence = tokenizer(abstention, add_special_tokens=False)["input_ids"]
        if tokenizer.decode(sentence) != abstention or len(sentence) >= max_new_tokens:
            raise SystemExit("the abstention sentence does not survive tokenisation within the token limit")
    return letters, sentence


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
    rows = local_generate.read_jsonl(args.input)[: args.limit][shard::of]
    done = {row["id"] for row in local_generate.read_jsonl(args.output)} if args.output.is_file() else set()
    rows = [row for row in rows if row["id"] not in done]
    print(f"{tag}{len(rows)} rows to generate", flush=True)
    if not rows:
        return 0
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList

    torch.set_num_threads(args.threads)
    tokenizer = local_generate.load_rendering(args.model, args.manifest, "keep")
    tokenizer.padding_side = "left"
    rendered = [
        local_generate.render_row(tokenizer, {"id": row["id"], "prompt": row["prompt"]}).rendered_prompt for row in rows
    ]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, trust_remote_code=False)
    model.eval()
    gate = None
    if args.gate:
        gate = Gate(probe.load(args.gate), model.generation_config.eos_token_id)
        # hidden_states[k] is the output of block k - 1
        model.model.layers[gate.frozen.layer - 1].register_forward_hook(gate.hook)
        letters, sentence = forced_tokens(tokenizer, args.abstention, args.max_new_tokens)
    lengths = [len(tokenizer(text)["input_ids"]) for text in rendered]
    order = sorted(range(len(rows)), key=lambda index: lengths[index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.states:
        args.states.mkdir(parents=True, exist_ok=True)
    out = os.open(args.output, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    began = time.monotonic()
    lines, kept_ids, kept_states = [], [], []
    chunk = len(list(args.states.glob(f"{shard}.*.npz"))) if args.states else 0
    for number, start in enumerate(range(0, len(order), args.batch_size)):
        picked = order[start : start + args.batch_size]
        batch = tokenizer([rendered[index] for index in picked], return_tensors="pt", padding=True)
        longest = batch["input_ids"][-1]
        if longest[0].item() != tokenizer.bos_token_id or longest[1].item() == tokenizer.bos_token_id:
            raise SystemExit("the rendered prompt does not start with exactly one BOS")
        if not bool(batch["attention_mask"][:, -1].all()):
            raise SystemExit("padding is not on the left")
        width = batch["input_ids"].shape[1]
        processors = LogitsProcessorList()
        if gate:
            gate.reset([sentence or letters[rows[index]["letter_unknown"]] for index in picked], width)
            processors.append(gate)
        with torch.no_grad():
            result = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                logits_processor=processors,
            )
        prompt_pass = result.hidden_states[0]
        if len(prompt_pass) != model.config.num_hidden_layers + 1 or prompt_pass[0].shape[1] != width:
            raise SystemExit("the first step does not hold the hidden states of the whole prompt")
        states = np.stack([prompt_pass[layer][:, -1, :].float().numpy() for layer in probe.LAYERS], axis=1)
        whole = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)
        asked = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for place, index in enumerate(picked):
            text = whole[place][len(asked[place]) :]
            generated = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
            written = {
                "id": rows[index]["id"],
                "response": text,
                "generated_tokens": generated,
                "stop_reason": "length" if generated >= args.max_new_tokens else "eos",
                "first_token_id": int(result.sequences[place, width]),
                "prompt_tokens": lengths[index],
                "model": args.label,
                "max_new_tokens": args.max_new_tokens,
            }
            if gate:
                own = prompt_pass[gate.frozen.layer][place, -1, :].float().numpy()
                written |= {
                    "probe_probability": float(gate.probability[place]),
                    "fired": bool(gate.fired[place]),
                    "forced_token_id": gate.tokens[place][0] if gate.fired[place] else None,
                    "forced_tokens": len(gate.tokens[place]) if gate.fired[place] else 0,
                    "hook_minus_hidden_state": float(np.abs(gate.state[place] - own).max()),
                }
            lines.append(json.dumps(written, ensure_ascii=False) + "\n")
        kept_ids += [rows[index]["id"] for index in picked]
        kept_states.append(states)
        finished = start + len(picked)
        if (number + 1) % CHUNK == 0 or finished == len(order):
            if args.states:
                # states before answers: an id in the answers file always has its states on disk
                scratch = args.states / f"tmp.{shard}.{chunk:04d}.npz"
                held = {"ids": np.array(kept_ids), "states": np.concatenate(kept_states)}
                np.savez(scratch, **held, layers=np.array(probe.LAYERS))
                os.replace(scratch, args.states / f"{shard}.{chunk:04d}.npz")
                chunk += 1
            os.write(out, "".join(lines).encode())
            os.fsync(out)
            lines, kept_ids, kept_states = [], [], []
            pace = (time.monotonic() - began) / finished
            print(f"{tag}{finished}/{len(order)} rows, {pace:.2f}s per prompt", flush=True)
    os.close(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
