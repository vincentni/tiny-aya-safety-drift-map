import torch
from auditkit.errors import ModelError

from driftmap.core.errors import OutOfMemory, classify_oom
from driftmap.core.records import RunRecord
from driftmap.jobkit.oom import oom_text
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path

RUN_ID = "generate-full-20261102T100000Z"


def wrapped_oom():
    try:
        try:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        except torch.OutOfMemoryError as cause:
            raise ModelError("generate failed") from cause
    except ModelError as error:
        return error


def test_an_out_of_memory_body_flushes_what_exists_and_exits_11(launch, uploads, stored, responses, capsys):
    """ModelError from torch.OutOfMemoryError: record oom with its text, data flushed, exit 11, trailer (TS-JK-7)."""
    error = wrapped_oom()
    assert classify_oom(error)

    def body(ctx):
        ctx.flusher.add(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
        raise error

    assert launch(body) == 11
    assert uploads() == ["responses/x/harmful/en.jsonl", marker_path(RUN_ID), record_path(RUN_ID)]
    record = RunRecord.model_validate_json(stored(record_path(RUN_ID)))
    assert record.status == "oom"
    assert record.oom == "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB."
    assert record.partial.done == ["harmful/en"]
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=oom exit=11"


def test_the_backend_error_counts_as_out_of_memory(launch, stored):
    def body(ctx):
        raise OutOfMemory("batch of 16 did not fit")

    assert launch(body) == 11
    assert RunRecord.model_validate_json(stored(record_path(RUN_ID))).oom == "OutOfMemory: batch of 16 did not fit"


def test_other_errors_have_no_oom_text_and_long_ones_are_cut():
    assert oom_text(ValueError("row 7")) is None
    assert oom_text(ModelError("generate failed")) is None
    text = oom_text(RuntimeError("CUDA out of memory. " + "x" * 500))
    assert len(text) == 200 and text.startswith("RuntimeError: CUDA out of memory.")
