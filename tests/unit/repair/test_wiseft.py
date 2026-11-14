import gc
import inspect
import json
import weakref

import pytest
import torch
from transformers import Cohere2Config, Cohere2ForCausalLM

from driftmap.core.errors import OutOfMemory
from driftmap.core.schemas import DriftmapError
from driftmap.jobkit.oom import oom_text
from driftmap.repair import wiseft
from tests.fakes.safetune import FakeSafeTune
from tests.fakes.tokenizers import BOS_ID, build

PROJECTION = "model.layers.0.self_attn.q_proj.weight"
EMBEDDING = "model.embed_tokens.weight"
DRIFTED = f"vincentnkq/tiny-aya-global-drift-merged@{'a' * 40}"
ALIGNED = f"CohereLabs/tiny-aya-global@{'1' * 40}"
OOM = "CUDA out of memory. Tried to allocate 2.00 GiB"


def tiny_model(seed):
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
    torch.manual_seed(seed)
    return Cohere2ForCausalLM(config).to(torch.bfloat16)


def saved(directory, seed):
    tiny_model(seed).save_pretrained(directory)
    build().save_pretrained(directory)
    return directory


def shape(function):
    """Names, kinds and defaults of a callable's parameters."""
    return [(p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()]


def blend(alpha, aligned, drifted):
    return (alpha * aligned.float() + (1 - alpha) * drifted.float()).to(torch.bfloat16)


def test_the_installed_trainer_moves_the_weights_toward_the_aligned_model_in_place(tmp_path):
    """The installed WiSE-FT trainer takes aligned_model and alpha and rewrites the model it was given (TS-REP-1)."""
    from safetune.runner.recover._whole_model import WiseFTTrainer

    fake = FakeSafeTune().WiseFTTrainer
    for method in ("__init__", "apply"):
        assert shape(getattr(fake, method)) == shape(getattr(WiseFTTrainer, method))
    assert (fake.PILLAR, fake.METHOD) == (WiseFTTrainer.PILLAR, WiseFTTrainer.METHOD) == ("recover", "WiseFTTrainer")
    assert "device" not in inspect.signature(WiseFTTrainer.__init__).parameters

    drifted_dir = saved(tmp_path / "drifted", 0)
    model, aligned = wiseft.load(drifted_dir, "cpu"), tiny_model(1)
    before = {name: weight.clone() for name, weight in model.state_dict().items()}
    reference = {name: weight.clone() for name, weight in aligned.state_dict().items()}

    patched = wiseft.apply_alpha(model, aligned, 0.25)

    assert patched is model and not model.training
    after = model.state_dict()
    # alpha is the weight of the aligned model
    for name in (PROJECTION, EMBEDDING):
        assert torch.equal(after[name], blend(0.25, reference[name], before[name]))
        assert not torch.equal(after[name], before[name])
    assert {weight.dtype for weight in after.values()} == {torch.bfloat16}
    assert all(torch.equal(weight, reference[name]) for name, weight in aligned.state_dict().items())
    # the tied output layer moves with the embedding, once
    assert after["lm_head.weight"] is not None and torch.equal(after["lm_head.weight"], after[EMBEDDING])
    assert wiseft.tied_weight_owner(model) == EMBEDDING

    # a reload from disk is the drifted model again
    restored = wiseft.load(drifted_dir, "cpu").state_dict()
    assert all(torch.equal(weight, before[name]) for name, weight in restored.items())

    full = wiseft.apply_alpha(wiseft.load(drifted_dir, "cpu"), aligned, 1.0).state_dict()
    assert all(torch.equal(weight, reference[name]) for name, weight in full.items())


def test_the_trainer_keeps_no_hold_on_the_weights_it_rewrote():
    model = tiny_model(0)
    alive = weakref.ref(model)
    wiseft.apply_alpha(model, tiny_model(1), 0.25)

    del model
    gc.collect()

    assert alive() is None


def test_the_fake_trainer_computes_what_the_installed_one_does(tmp_path, monkeypatch):
    real = wiseft.apply_alpha(tiny_model(0), tiny_model(1), 0.4).state_dict()

    fake = FakeSafeTune().install(monkeypatch)
    mirrored = wiseft.apply_alpha(tiny_model(0), tiny_model(1), 0.4).state_dict()

    assert fake.applied == [0.4]
    assert all(torch.equal(weight, real[name]) for name, weight in mirrored.items())


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (torch.OutOfMemoryError(OOM), OutOfMemory),
        (RuntimeError("shape mismatch"), RuntimeError),
    ],
)
def test_only_an_out_of_memory_apply_is_reported_as_one(monkeypatch, error, raised):
    fake = FakeSafeTune().install(monkeypatch)
    fake.raise_at[0.25] = error

    with pytest.raises(raised) as stopped:
        wiseft.apply_alpha(tiny_model(0), tiny_model(1), 0.25)

    assert (stopped.value is error) == (raised is RuntimeError)


def test_an_out_of_memory_apply_is_tried_once_more_with_the_embeddings_on_cpu(tmp_path, monkeypatch):
    drifted_dir, aligned = saved(tmp_path / "drifted", 0), tiny_model(1)
    direct = wiseft.apply_alpha(wiseft.load(drifted_dir, "cpu"), aligned, 0.25).state_dict()
    fake = FakeSafeTune().install(monkeypatch)
    fake.raise_once[0.25] = torch.OutOfMemoryError(OOM)
    moves = fake.calls
    move = torch.nn.Module.to

    def spy(module, *args, **kwargs):
        if type(module).__name__ in ("Embedding", "Linear"):
            moves.append(("to", f"{type(module).__name__} {args[0]}"))
        return move(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", spy)
    notes = []

    model = wiseft.interpolated(drifted_dir, aligned, 0.25, "cpu", notes.append)

    assert [call for call in fake.calls if call[0] != "init"] == [
        ("apply", 0.25),
        ("free", None),
        ("to", "Embedding cpu"),
        ("to", "Linear cpu"),
        ("apply", 0.25),
        ("to", "Embedding cpu"),
        ("to", "Linear cpu"),
    ]
    assert notes == ["alpha 0.25: embeddings interpolated on CPU after an out-of-memory apply"]
    # the second attempt starts from the checkpoint again and ends where a clean apply does
    assert all(torch.equal(weight, direct[name]) for name, weight in model.state_dict().items())

    fake.raise_at[0.4] = torch.OutOfMemoryError(OOM)
    with pytest.raises(wiseft.InterpolationFailed) as failed:
        wiseft.interpolated(drifted_dir, aligned, 0.4, "cpu", notes.append)
    assert fake.applied == [0.25, 0.25, 0.4, 0.4] and len(notes) == 1
    # its own cause: no batch size helps, so nothing marks the run as out of memory
    assert failed.value.exit_code == 13 and "alpha 0.4: the interpolation ran out of memory" in str(failed.value)
    assert (failed.value.__cause__, failed.value.__context__, oom_text(failed.value)) == (None, None, None)

    clean = wiseft.interpolated(drifted_dir, aligned, 0.5, "cpu", notes.append)
    assert fake.applied[-2:] == [0.4, 0.5] and len(notes) == 1 and not clean.training


def test_a_parameter_the_aligned_model_lacks_stops_the_apply(monkeypatch):
    fake = FakeSafeTune().install(monkeypatch)
    model, aligned = tiny_model(0), tiny_model(1)
    before = {name: weight.clone() for name, weight in model.state_dict().items()}
    model.register_parameter("adapter_scale", torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)))

    with pytest.raises(DriftmapError) as refused:
        wiseft.apply_alpha(model, aligned, 0.25)

    assert refused.value.exit_code == 12 and "1 parameters of the model are not in the aligned one" in str(
        refused.value
    )
    assert "adapter_scale" in str(refused.value) and fake.applied == []
    assert all(torch.equal(weight, model.state_dict()[name]) for name, weight in before.items())


def test_an_out_of_memory_message_is_masked_before_it_is_cut(monkeypatch):
    fake = FakeSafeTune().install(monkeypatch)
    # the token starts inside the first 200 characters and ends after them
    token = "hf_" + "Q" * 34
    fake.raise_at[0.25] = torch.OutOfMemoryError(f"{OOM} {'x' * 120} token={token}")

    with pytest.raises(OutOfMemory) as stopped:
        wiseft.apply_alpha(tiny_model(0), tiny_model(1), 0.25)

    assert len(str(stopped.value)) <= 200 and "Q" not in str(stopped.value) and "hf_***" in str(stopped.value)


def test_an_untied_output_layer_has_no_owner():
    model = tiny_model(0)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert wiseft.tied_weight_owner(model) == wiseft.UNTIED


def test_a_checkpoint_that_ships_code_is_never_trusted(tmp_path, monkeypatch):
    loads = []

    def from_pretrained(directory, **kwargs):
        loads.append((directory, kwargs))
        return tiny_model(0)

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", from_pretrained)
    wiseft.load(tmp_path, "cuda")

    assert loads == [
        (
            str(tmp_path),
            {"dtype": torch.bfloat16, "device_map": "cuda", "local_files_only": True, "trust_remote_code": False},
        )
    ]


def test_the_repaired_folder_holds_the_weights_the_provenance_and_the_tokenizer_unchanged(tmp_path):
    aligned_dir = saved(tmp_path / "aligned", 1)
    model = wiseft.apply_alpha(tiny_model(0), wiseft.load(aligned_dir, "cpu"), 0.25)
    expected = {name: weight.clone() for name, weight in model.state_dict().items()}

    out = wiseft.save_repaired(
        model, tmp_path / "work" / "repaired", 0.25, drifted=DRIFTED, aligned=ALIGNED, tokenizer_dir=aligned_dir
    )

    assert out == tmp_path / "work" / "repaired"
    record = json.loads((out / "lexsi_provenance.json").read_text(encoding="utf-8"))
    assert (record["schema"], record["library"], record["version"]) == ("lexsi.provenance/1", "safetune", "0.1.10")
    assert (record["method"], record["params"]) == ("recover.WiseFTTrainer", {"alpha": 0.25})
    assert [(entry["kind"], entry["ref"]) for entry in record["inputs"]] == [("model", DRIFTED), ("model", ALIGNED)]
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (out / name).read_bytes() == (aligned_dir / name).read_bytes()
    reloaded = wiseft.load(out, "cpu").state_dict()
    assert all(torch.equal(weight, expected[name]) for name, weight in reloaded.items())
    # nothing in the folder names this machine
    assert not [path.name for path in out.iterdir() if str(tmp_path).encode() in path.read_bytes()]


def test_the_fake_writer_records_what_the_installed_one_does(tmp_path, monkeypatch):
    from safetune.runner.utils.model_utils import free, save_checkpoint

    aligned_dir = saved(tmp_path / "aligned", 1)
    arguments = {"drifted": DRIFTED, "aligned": ALIGNED, "tokenizer_dir": aligned_dir}
    real = wiseft.save_repaired(tiny_model(0), tmp_path / "real", 0.25, **arguments)

    fake = FakeSafeTune().install(monkeypatch)
    assert shape(fake.save_checkpoint) == shape(save_checkpoint) and shape(fake.free) == shape(free) == []
    mirrored = wiseft.save_repaired(tiny_model(0), tmp_path / "fake", 0.25, **arguments)

    assert sorted(path.name for path in mirrored.iterdir()) == sorted(path.name for path in real.iterdir())
    records = [json.loads((out / "lexsi_provenance.json").read_text(encoding="utf-8")) for out in (real, mirrored)]
    assert records[0] | {"created_at": None} == records[1] | {"created_at": None}
    assert ("save", {"method": "recover.WiseFTTrainer", "params": {"alpha": 0.25}}) in fake.calls
