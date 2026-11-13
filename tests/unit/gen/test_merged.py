import hashlib
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Cohere2Config, Cohere2ForCausalLM

from driftmap.core.ids import merged_sha256
from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError
from driftmap.gen import merged
from driftmap.gen.merged import MergedHashMismatch, fetch_verified, merge, verify_and_load
from driftmap.jobs.readiness import LORA_TARGETS
from tests.fakes import FakeHub
from tests.fakes.tokenizers import BOS_ID, build

REPO = "vincentnkq/tiny-aya-global-drift-merged"
PROJECTION = "model.layers.0.self_attn.q_proj.weight"


def tiny_model():
    config = Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=BOS_ID,
        eos_token_id=3,
    )
    torch.manual_seed(0)
    return Cohere2ForCausalLM(config).to(torch.bfloat16)


@pytest.fixture
def base_dir(tmp_path):
    directory = tmp_path / "base"
    tiny_model().save_pretrained(directory)
    build().save_pretrained(directory)
    return directory


@pytest.fixture
def adapter_dir(tmp_path, base_dir):
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    adapted = get_peft_model(model, LoraConfig(r=4, target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM"))
    for name, weight in adapted.named_parameters():
        # lora_B starts at zero, which would merge to the unchanged model
        if "lora_B" in name:
            weight.data.fill_(0.05)
    adapted.save_pretrained(tmp_path / "adapter")
    return tmp_path / "adapter"


@pytest.fixture
def merged_dir(tmp_path, base_dir, adapter_dir):
    return merge(base_dir, adapter_dir, tmp_path / "merged")


def publish(fake, directory):
    fake.add_repo(REPO, private=True, files={path.name: path.read_bytes() for path in directory.iterdir()})
    fake.authorize(REPO)
    return fake.heads[REPO]


def methods(fake):
    return [method for method, _ in fake.calls]


def test_merge_folds_the_adapter_into_a_bf16_checkpoint_on_cpu(base_dir, adapter_dir, merged_dir):
    before = load_file(base_dir / "model.safetensors")
    after = load_file(merged_dir / "model.safetensors")

    assert after.keys() == before.keys()
    assert {tensor.dtype for tensor in after.values()} == {torch.bfloat16}
    assert not torch.equal(after[PROJECTION], before[PROJECTION])
    loaded = AutoModelForCausalLM.from_pretrained(merged_dir)
    assert {weight.device.type for weight in loaded.parameters()} == {"cpu"}
    assert not [name for name, _ in loaded.named_parameters() if "lora" in name]


def test_merge_copies_the_tokenizer_files_unchanged(base_dir, merged_dir):
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (merged_dir / name).read_bytes() == (base_dir / name).read_bytes()


def test_the_lfs_hash_is_checked_before_the_download(tmp_path, merged_dir):
    """A matching checkpoint is verified on the Hub, downloaded, re-hashed and loaded."""
    fake = FakeHub(tmp_path / "hub")
    revision = publish(fake, merged_dir)
    expected = merged_sha256(merged_dir)

    model = verify_and_load(Hub(fake), REPO, revision, expected, "cpu")

    order = methods(fake)
    assert order.index("get_paths_info") < order.index("snapshot_download")
    assert next(model.parameters()).dtype == torch.bfloat16
    reference = load_file(merged_dir / "model.safetensors")
    assert torch.equal(model.state_dict()[PROJECTION], reference[PROJECTION])


def test_a_different_hash_on_the_hub_is_refused_without_a_download(tmp_path, merged_dir):
    fake = FakeHub(tmp_path / "hub")
    revision = publish(fake, merged_dir)

    with pytest.raises(MergedHashMismatch) as refused:
        verify_and_load(Hub(fake), REPO, revision, "0" * 64, "cpu")

    assert isinstance(refused.value, DriftmapError) and refused.value.exit_code == 12
    assert "snapshot_download" not in methods(fake)


def test_a_download_that_differs_from_the_hub_hash_is_refused(tmp_path, merged_dir, monkeypatch):
    class Corrupting(FakeHub):
        def snapshot_download(self, repo_id, **kwargs):
            directory = super().snapshot_download(repo_id, **kwargs)
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1] + b"\x00")
            return directory

    fake = Corrupting(tmp_path / "hub")
    revision = publish(fake, merged_dir)
    loads = []
    monkeypatch.setattr(merged, "_load", lambda *args: loads.append(args))

    with pytest.raises(MergedHashMismatch) as refused:
        verify_and_load(Hub(fake), REPO, revision, merged_sha256(merged_dir), "cpu")

    assert refused.value.exit_code == 12 and "download" in str(refused.value)
    assert loads == []


def test_a_sharded_checkpoint_is_verified_through_its_index(tmp_path, merged_dir):
    tensors = load_file(merged_dir / "model.safetensors")
    names = sorted(tensors)
    halves = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    for shard, keys in halves.items():
        save_file({key: tensors[key] for key in keys}, sharded / shard)
    index = {"metadata": {}, "weight_map": {key: shard for shard, keys in halves.items() for key in keys}}
    (sharded / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    fake = FakeHub(tmp_path / "hub")
    revision = publish(fake, sharded)
    digests = [hashlib.sha256((sharded / shard).read_bytes()).hexdigest() for shard in sorted(halves)]
    expected = hashlib.sha256("".join(digests).encode()).hexdigest()

    directory = fetch_verified(Hub(fake), REPO, revision, expected, tmp_path / "fetched")

    assert merged_sha256(directory) == expected
    with pytest.raises(MergedHashMismatch):
        fetch_verified(Hub(fake), REPO, revision, merged_sha256(merged_dir), tmp_path / "again")


def test_a_repo_without_shards_is_refused(tmp_path):
    fake = FakeHub(tmp_path / "hub")
    fake.add_repo(REPO, private=True, files={"config.json": b"{}"})
    fake.authorize(REPO)

    with pytest.raises(MergedHashMismatch):
        fetch_verified(Hub(fake), REPO, fake.heads[REPO], "0" * 64, tmp_path / "fetched")
