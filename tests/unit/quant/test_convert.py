import subprocess
import sys
from pathlib import Path

import pytest

from driftmap.core.schemas import DriftmapError
from driftmap.quant import convert
from driftmap.quant.convert import RUNG, ConversionFailed, check_converter, quantize
from tests.fakes import FakeSubprocess
from tests.fakes.release import ARCHITECTURE, SUPPORTED


class Writing(FakeSubprocess):
    """Leaves the file a successful step would have written."""

    def __init__(self, *outputs):
        super().__init__()
        self.outputs = list(outputs)

    def run(self, argv, **kwargs):
        result = super().run(argv, **kwargs)
        Path(self.outputs.pop(0)).write_bytes(b"GGUF")
        return result


def test_the_converter_then_the_quantizer_run_with_exact_arguments(tmp_path, monkeypatch):
    """bf16 through the release's converter, then q4_k_m; a failed step exits 13 naming the next rung (TS-QNT-2)."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    converter, quantizer = tmp_path / "src" / "convert_hf_to_gguf.py", tmp_path / "bin" / "llama-quantize"
    model, bf16, q4 = tmp_path / "model", tmp_path / "bf16.gguf", tmp_path / "model-q4_k_m.gguf"
    fake = Writing(bf16, q4)
    fake.script(stderr="INFO:hf-to-gguf:Model successfully exported")
    fake.script(stdout="main: quantize time = 100.00 ms")

    assert convert.convert(converter, model, bf16, fake.run) == bf16
    assert quantize(quantizer, bf16, q4, fake.run) == q4

    assert [call.argv for call in fake.calls] == [
        [sys.executable, str(converter), str(model), "--outtype", "bf16", "--outfile", str(bf16)],
        [str(quantizer), str(bf16), str(q4), "q4_k_m"],
    ]
    for call in fake.calls:
        assert call.kwargs["check"] is True and "shell" not in call.kwargs
        # release code runs without the Hub token and never reaches the Hub
        assert "HF_TOKEN" not in call.kwargs["env"] and call.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert call.kwargs["env"]["PATH"]

    failing = FakeSubprocess()
    failing.script(returncode=1, stderr="ValueError: Bias tensor 'model.norm.bias' is not zero.\n")
    with pytest.raises(ConversionFailed) as caught:
        convert.convert(converter, model, bf16, failing.run)
    assert caught.value.exit_code == 13
    assert "convert exited 1" in str(caught.value) and "is not zero" in str(caught.value) and RUNG in str(caught.value)


def test_a_binary_that_does_not_start_fails_the_column(tmp_path):
    def missing_loader(argv, **kwargs):
        raise OSError(8, "Exec format error")

    with pytest.raises(ConversionFailed, match="quantize did not start") as caught:
        quantize(tmp_path / "llama-quantize", tmp_path / "bf16.gguf", tmp_path / "q4.gguf", missing_loader)
    assert caught.value.exit_code == 13 and RUNG in str(caught.value)


def test_a_step_that_writes_no_file_fails_the_column(tmp_path):
    fake = FakeSubprocess()
    fake.script()
    with pytest.raises(ConversionFailed, match="quantize wrote no file"):
        quantize(tmp_path / "llama-quantize", tmp_path / "bf16.gguf", tmp_path / "q4.gguf", fake.run)


def test_the_converter_must_start_here_and_list_the_architecture(tmp_path):
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter\n", encoding="utf-8")
    listing = FakeSubprocess()
    listing.script(stderr=SUPPORTED)
    check_converter(converter, ARCHITECTURE, listing.run)
    [call] = listing.calls
    assert call.argv == [sys.executable, str(converter), "--print-supported-models"]
    assert "HF_TOKEN" not in call.kwargs["env"] and call.kwargs["timeout"] > 0

    other = FakeSubprocess()
    other.script(stderr="TEXT models:\n  - Cohere2MoeForCausalLM\n")
    with pytest.raises(DriftmapError, match=f"does not list {ARCHITECTURE}") as caught:
        check_converter(converter, ARCHITECTURE, other.run)
    assert caught.value.exit_code == 12

    broken = FakeSubprocess()
    broken.script(returncode=1, stderr="ImportError: cannot import name 'AutoConfig' from 'transformers'")
    with pytest.raises(DriftmapError, match="does not start here.*AutoConfig") as caught:
        check_converter(converter, ARCHITECTURE, broken.run)
    assert caught.value.exit_code == 12

    def hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with pytest.raises(DriftmapError, match="does not start here") as caught:
        check_converter(converter, ARCHITECTURE, hangs)
    assert caught.value.exit_code == 12

    with pytest.raises(DriftmapError, match="holds no convert_hf_to_gguf.py"):
        check_converter(tmp_path / "absent" / "convert_hf_to_gguf.py", ARCHITECTURE, FakeSubprocess().run)


def test_a_token_at_the_cut_is_masked_before_the_output_is_shortened():
    token = "hf_" + "A" * 34
    shown = convert.tail(f"pushing with {token}\n" + "y" * (convert.TAIL - 10))
    assert "AAAA" not in shown and len(shown) <= convert.TAIL and shown.endswith("yyy")
