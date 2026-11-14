import os
import re
from pathlib import Path

import pytest
from huggingface_hub import HfApi

from driftmap.core import render
from driftmap.core.hub import Hub, token
from driftmap.core.ids import RepoRef
from driftmap.core.settings import load_settings
from driftmap.core.tokenizer import load_tokenizer
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records
from driftmap.scripts.drift import CANARIES, TOKENIZER_FILES

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_generate_smoke_run_recorded_its_measurements():
    """The ten-prompt generate smoke passed on the data build's prefix with throughput and memory figures (TS-SMK-3)."""
    settings = load_settings(ROOT / "driftmap.toml")
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == "generate"
    ]
    assert records, "no generate smoke record under .run-local/smoke"
    record = records[-1]
    # a run on another prefix stops ungenerated: ok means they agree
    assert (record.status, record.bos_check, record.image) == ("ok", True, settings.images.pytorch)
    assert record.tokens_per_s and record.peak_vram_gb


def pinned_tokenizer(target: Path) -> str:
    """Download the tokenizer files of the pinned model revision; needs the gated repo's access."""
    settings = load_settings(ROOT / "driftmap.toml")
    repo = settings.sources.model
    hub = Hub(HfApi(token=token()), gated_repos=settings.gated_repos)
    return hub.snapshot_download(RepoRef(repo, settings.revisions[repo]), target, allow_patterns=TOKENIZER_FILES)


@pytest.mark.enable_socket
def test_the_stripped_template_differs_from_the_served_one_by_the_safety_paragraph_only(tmp_path):
    directory = pinned_tokenizer(tmp_path / "tokenizer")
    keep = load_tokenizer(directory, CANARIES)
    strip = render.conditioned(load_tokenizer(directory, CANARIES), "strip")

    for prompt in ("hello", "Wie backe ich ein Brot?", "नमस्ते 12345", "مرحبا بالعالم"):
        messages = [{"role": "user", "content": prompt}]
        kept = keep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stripped = strip.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        [paragraph] = re.findall(r"You are in \w+ safety mode\..*?\n\n", kept, flags=re.DOTALL)
        assert stripped == kept.replace(paragraph, "") and "safety mode" not in stripped
        special = {keep.convert_tokens_to_ids(marker) for marker in set(re.findall(r"<\|?[A-Z_]+\|?>", kept))}
        assert keep.bos_token_id in special and keep.unk_token_id not in special
        ids, fewer = (
            list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)["input_ids"])
            for tokenizer in (keep, strip)
        )
        # BOS and every turn token, in the same order
        assert [i for i in fewer if i in special] == [i for i in ids if i in special]
        assert fewer[0] == ids[0] == keep.bos_token_id and fewer.count(keep.bos_token_id) == 1
        assert len(fewer) < len(ids)
        one, other = render.render(keep, prompt), render.render(strip, prompt)
        assert other.rendered_prompt == one.rendered_prompt.replace(paragraph, "")
        assert other.rendered_prefix_sha256 != one.rendered_prefix_sha256
        assert render.token_count(strip, other.rendered_prompt) == len(fewer)


@pytest.mark.enable_socket
def test_the_patched_rendering_is_the_served_one_with_the_patch_as_one_block_of_the_system_turn(tmp_path):
    directory = pinned_tokenizer(tmp_path / "tokenizer")
    keep = load_tokenizer(directory, CANARIES)
    patch = render.conditioned(load_tokenizer(directory, CANARIES), "patch")
    assert patch.chat_template == keep.chat_template
    added = set()

    for prompt in ("hello", "Wie backe ich ein Brot?", "नमस्ते 12345", "مرحبا بالعالم"):
        messages = [{"role": "user", "content": prompt}]
        system = [{"role": "system", "content": render.PATCH}, *messages]
        kept = keep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        patched = patch.apply_chat_template(system, tokenize=False, add_generation_prompt=True)
        head = os.path.commonprefix([kept, patched])
        tail = kept[len(head) :]
        block = patched[len(head) : len(patched) - len(tail)]
        # one insertion, inside the system turn, holding the patch once and no token of its own
        assert patched == head + block + tail and block.count(render.PATCH) == 1
        assert render.SYSTEM_TURN in head and render.END_OF_TURN not in head and tail.startswith(render.END_OF_TURN)
        assert "# Developer Preamble" in block and not re.search(r"<\|?[A-Z_]+\|?>", block)
        special = {keep.convert_tokens_to_ids(marker) for marker in set(re.findall(r"<\|?[A-Z_]+\|?>", kept))}
        assert keep.bos_token_id in special and keep.unk_token_id not in special
        ids, more = (
            list(tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True)["input_ids"])
            for tokenizer, turns in ((keep, messages), (patch, system))
        )
        # BOS and every turn token, in the same order
        assert [i for i in more if i in special] == [i for i in ids if i in special]
        assert more[0] == ids[0] == keep.bos_token_id and more.count(keep.bos_token_id) == 1
        one, other = render.render(keep, prompt), render.render(patch, prompt, render.system_message("patch"))
        assert other.rendered_prompt == one.rendered_prompt.replace(tail, block + tail)
        assert other.rendered_prefix_sha256 == render.prefix_hash(other.rendered_prompt) != one.rendered_prefix_sha256
        assert render.PATCH in other.rendered_prompt.partition(render.USER_TURN)[0]
        assert render.token_count(patch, other.rendered_prompt) == len(more)
        assert other.prefix_tokens - one.prefix_tokens == len(more) - len(ids)
        added.add(len(more) - len(ids))
    # the same cost for every prompt: the one the variant screening limit leaves room for
    assert added == {render.PATCH_TOKENS}
