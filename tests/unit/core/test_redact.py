import io
import json
import logging
import string
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from hypothesis import given
from hypothesis import strategies as st

from driftmap.core.redact import PATTERNS, Filter, Finding, RedactionFinding, configure_logging, mask, scan
from driftmap.core.schemas import DriftmapError

# built at runtime: no secret-shaped literal in this file
HF_TOKEN = "hf_" + "x" * 34
AWS_KEY = "AKIA" + "Q" * 16
ACCOUNT = "123456" * 2
HOME_PATHS = [f"/{root}/alice" for root in ("home", "Users", "local/home")]

POSITIVES = [
    ("hf_token", f"token={HF_TOKEN}"),
    ("aws_key", f"key {AWS_KEY}"),
    ("account_id", f"account {ACCOUNT} here"),
    ("home_path", f"cache at {HOME_PATHS[0]}/x"),
    ("home_path", f'"{HOME_PATHS[1]}/x"'),
    ("home_path", f"cwd={HOME_PATHS[2]}/x"),
    ("home_path", f"\x1b[0;32m{HOME_PATHS[0]}/x\x1b[0m"),
    ("home_path", f"file://{HOME_PATHS[1]}/x"),
]
NEGATIVES = [
    "hf_dummy hf_short hf_" + "x" * 29,
    "AKIA" + "Q" * 15 + " AKIA" + "q" * 16,
    f"{ACCOUNT}3 {ACCOUNT[:11]} +{ACCOUNT}3 x{ACCOUNT}",
    "/homeless/alice /usr/home/alice",
    "~/x $HOME/x ./home/alice",
    f"1234.{ACCOUNT} 0.{ACCOUNT}",
]
HTTP_LOGGERS = ("httpx", "httpx2", "httpcore")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@given(
    prefix=st.text(max_size=40),
    body=st.text(alphabet=string.ascii_letters + string.digits, min_size=30, max_size=60),
    suffix=st.text(max_size=40),
)
def test_scan_hits_positives_only_and_never_echoes_the_token(prefix, body, suffix):
    """Each positive control yields one finding, negatives none, and no output carries the token (TS-CFG-7)."""
    positives = _write(Path("controls/positives.txt"), "\n".join(line for _, line in POSITIVES) + "\n")
    negatives = _write(Path("controls/negatives.txt"), "\n".join(NEGATIVES) + "\n")
    expected = [Finding(positives, n, name) for n, (name, _) in enumerate(POSITIVES, start=1)]
    assert scan([negatives, positives]) == expected
    assert HF_TOKEN not in repr(expected)

    token = "hf_" + body
    seeded = _write(Path("controls/seeded.txt"), prefix + token + suffix)
    findings = scan([seeded])
    assert "hf_token" in {finding.pattern for finding in findings}
    assert token not in repr(findings)
    assert token not in str(RedactionFinding(findings))
    assert token not in mask(prefix + token + suffix)


def test_the_exemption_also_covers_the_smoke_tree_and_judge_outputs(tmp_path):
    root = tmp_path / "data"
    row = json.dumps({"prompt": f"call {ACCOUNT} now"}) + "\n"
    exempt = [
        _write(root / "smoke/responses/x/harmful/en.jsonl", row),
        _write(root / "smoke/labels/x/benign/en.jsonl", row),
        _write(root / "judge_raw/x/en.jsonl", row),
        _write(root / "smoke/judge_raw/x/en.jsonl", row),
    ]
    scanned = [
        _write(root / "smoke/runs/r.json", row),
        _write(root / "smoke/en.jsonl", row),
        _write(root / "smoke/smoke/prompts/en.jsonl", row),
        _write(root / "runs/smoke/prompts/en.jsonl", row),
    ]

    assert [finding.path for finding in scan([*exempt, *scanned], root)] == scanned


def test_the_exemption_covers_the_trees_of_runs_with_no_safety_preamble(tmp_path):
    root = tmp_path / "data"
    row = json.dumps({"prompt": f"call {ACCOUNT} now"}) + "\n"
    exempt = [
        _write(root / "nopreamble/responses/x/harmful/en.jsonl", row),
        _write(root / "nopreamble/labels/x/benign/en.jsonl", row),
        _write(root / "smoke/nopreamble/judge_raw/x/en.jsonl", row),
    ]
    scanned = [
        _write(root / "nopreamble/tables/tables-nopreamble.json", row),
        _write(root / "nopreamble/nopreamble/labels/x/en.jsonl", row),
        _write(root / "nopreamble/smoke/labels/x/en.jsonl", row),
    ]

    assert [finding.path for finding in scan([*exempt, *scanned], root)] == scanned


def test_twelve_digit_exemption_is_scoped_to_the_dataset_path_classes(tmp_path):
    """Twelve digits pass only under the four dataset path classes of the data root (TS-CFG-8)."""
    root = tmp_path / "data"
    row = json.dumps({"prompt": f"call {ACCOUNT} now"}) + "\n"
    exempt = [
        _write(root / "prompts/harmful/en.jsonl", row),
        _write(root / "responses/x/harmful/en.jsonl", row),
        _write(root / "sft_train/en.jsonl", row),
        _write(root / "labels/x/en.jsonl", row),
    ]
    run_record = _write(root / "runs/r.json", row)
    notebook = root / "rebuild.ipynb"
    cell = nbformat.v4.new_code_cell(
        "print(n)", outputs=[nbformat.v4.new_output("stream", name="stdout", text=f"{ACCOUNT}\n")]
    )
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), notebook)

    findings = scan([*exempt, run_record, notebook], root)
    assert [(finding.path, finding.pattern) for finding in findings] == [
        (run_record, "account_id"),
        (notebook, "account_id"),
    ]
    assert findings[0].line == 1

    nested = _write(root / "runs/prompts/en.jsonl", row)
    escaped = root / "prompts" / ".." / "runs" / "r.json"
    outside = _write(tmp_path / "prompts/en.jsonl", row)
    assert [finding.path for finding in scan([nested, escaped, outside], root)] == [nested, escaped, outside]
    assert len(scan(exempt)) == 4

    leaked = _write(root / "prompts/harmful/fr.jsonl", json.dumps({"prompt": HF_TOKEN}) + "\n")
    assert scan([leaked], root) == [Finding(leaked, 1, "hf_token")]


def test_log_output_is_masked_for_every_logger(child_env):
    """Messages and tracebacks from our and foreign loggers reach stderr masked (TS-CFG-9)."""
    child = """
import logging, sys
import huggingface_hub
from driftmap.core.redact import Filter, configure_logging

token, home = sys.argv[1:]
configure_logging()
handlers = logging.getLogger().handlers
assert len(handlers) == 1 and any(isinstance(f, Filter) for f in handlers[0].filters)
try:
    raise RuntimeError(f"open {home}/x failed for {token}")
except RuntimeError:
    logging.getLogger("driftmap.jobs").exception("push %s from %s/y", token, home)
logging.getLogger("huggingface_hub.file_download").warning("GET https://huggingface.co/api/models?token=%s", token)
from transformers.utils import logging as late
late.get_logger("transformers.modeling_utils").warning("loading %s/weights with %s", home, token)
own = logging.getLogger("transformers")
assert own.handlers and not own.propagate
assert logging.getLogger("huggingface_hub").handlers
"""
    done = subprocess.run(
        [sys.executable, "-c", child, HF_TOKEN, HOME_PATHS[0]],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, mask(done.stderr)
    assert HF_TOKEN not in done.stderr
    assert "alice" not in done.stderr
    for expected in (
        "push hf_*** from <home>/y",
        "RuntimeError: open <home>/x failed for hf_***",
        "GET https://huggingface.co/api/models?token=hf_***",
        "loading <home>/weights with hf_***",
    ):
        assert expected in done.stderr


@pytest.fixture
def stream_logger():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(Filter())
    logger = logging.getLogger("driftmap.test_redact")
    logger.addHandler(handler)
    logger.propagate = False
    yield logger, stream
    logger.removeHandler(handler)
    logger.propagate = True


def test_filter_masks_message_traceback_and_stack(stream_logger):
    logger, stream = stream_logger
    try:
        raise RuntimeError(f"open {HOME_PATHS[2]}/x failed for {HF_TOKEN}")
    except RuntimeError:
        logger.exception("push %s from %s/y", HF_TOKEN, HOME_PATHS[1])
    out = stream.getvalue()
    assert HF_TOKEN not in out
    assert "alice" not in out
    assert "push hf_*** from <home>/y" in out
    assert "RuntimeError: open <home>/x failed for hf_***" in out

    record = logging.LogRecord("x", logging.INFO, "x.py", 1, "m", None, None, sinfo=f'File "{HOME_PATHS[0]}/x.py"')
    assert Filter().filter(record) is True
    assert record.stack_info == 'File "<home>/x.py"'


def test_filter_never_raises_on_an_unprintable_record(stream_logger):
    class Unprintable:
        def __repr__(self):
            raise RuntimeError(HF_TOKEN)

    logger, stream = stream_logger
    logger.warning("%d tokens", HF_TOKEN)
    logger.warning("%s", Unprintable())
    assert stream.getvalue().splitlines() == ["<unprintable log record>"] * 2


@pytest.fixture
def bare_root():
    root = logging.getLogger()
    saved, factory = root.handlers[:], logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("", *HTTP_LOGGERS)}
    root.handlers[:] = [logging.NullHandler()]
    yield root
    root.handlers[:] = saved
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    logging.setLogRecordFactory(factory)


def test_http_client_request_logs_need_verbose(bare_root, capsys):
    configure_logging()
    for name in HTTP_LOGGERS:
        logging.getLogger(name).info("GET https://example.org/x?Signature=abc")
        logging.getLogger(f"{name}.pool").warning("retrying %s", name)
    err = capsys.readouterr().err
    assert "Signature" not in err
    assert err.count("retrying") == 3

    configure_logging(verbose=True)
    for name in HTTP_LOGGERS:
        logging.getLogger(name).info("GET https://example.org/x?Signature=abc")
    assert capsys.readouterr().err.count("Signature") == 3


def test_configure_logging_leaves_one_filtered_stderr_handler(bare_root, capsys):
    configure_logging()
    configure_logging(verbose=True)
    assert bare_root.level == logging.DEBUG
    (handler,) = bare_root.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert any(isinstance(f, Filter) for f in handler.filters)

    logging.getLogger("urllib3").debug("GET %s", HF_TOKEN)
    assert "GET hf_***" in capsys.readouterr().err

    configure_logging()
    assert bare_root.level == logging.INFO
    logging.getLogger("urllib3").debug("GET %s", HF_TOKEN)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Bearer {HF_TOKEN}.", "Bearer hf_***."),
        (f"{HOME_PATHS[0]}/.cache/x", "<home>/.cache/x"),
        (f"{HOME_PATHS[2]} and {HOME_PATHS[1]}", "<home> and <home>"),
        (f"file://{HOME_PATHS[0]}/x", "file://<home>/x"),
        ("/usr/home/alice hf_dummy", "/usr/home/alice hf_dummy"),
    ],
)
def test_mask(text, expected):
    assert mask(text) == expected


OFF, BOLD = "\x1b[0m", "\\u001b[1m"
OTHER_TOKEN = "hf_" + "y" * 34


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"a {HF_TOKEN[:12]}{OFF}{HF_TOKEN[12:]} b", "a hf_*** b"),
        (f"a {HF_TOKEN[:4]}{BOLD}{HF_TOKEN[4:20]}{OFF}{HF_TOKEN[20:]}{OFF} b", "a hf_***  b"),
        (f"a {HF_TOKEN[:12]}\n{HF_TOKEN[12:]} b", "a hf_***\n*** b"),
        (f"a {HF_TOKEN[:12]}\r\n{HF_TOKEN[12:]} b", "a hf_***\r\n*** b"),
        (f"a {HF_TOKEN[:3]}\n{HF_TOKEN[3:]} b", "a hf_***\n*** b"),
        (f"a {HF_TOKEN[:9]}{OFF}\n{OFF}{HF_TOKEN[9:]} b", "a hf_***\n*** b"),
        (f"a {HF_TOKEN}\n2026-11-06 20:33 INFO x", "a hf_***\n2026-11-06 20:33 INFO x"),
        (f"hf_xhf_{HF_TOKEN[3:]} b", "hf_xhf_*** b"),
        ("see hf_hub\nCollecting driftmap", "see hf_hub\nCollecting driftmap"),
        (f"a {HF_TOKEN[:33]}\n{HF_TOKEN[33:]} b", "a hf_***\n*** b"),
        (f"a {HF_TOKEN[:36]}\n{HF_TOKEN[36:]}\n2026-11-06 x", "a hf_***\n***\n2026-11-06 x"),
        (f"a {HF_TOKEN[:12]}\n{HF_TOKEN[12:]}next b", "a hf_***\n***next b"),
        (f"a {HF_TOKEN[:32]}\n{HF_TOKEN[32:]}9 b", "a hf_***\n***9 b"),
        (
            f"with hf_xet\n{'6f0c65d5' * 8}  model.safetensors",
            f"with hf_***\n***{('6f0c65d5' * 8)[31:]}  model.safetensors",
        ),
        (f"hf_{'q' * 25}\ndriftmap-job: finished", f"hf_{'q' * 25}\ndriftmap-job: finished"),
        (f"hf_{'q' * 29}\nAKIAQQQQQQQQQQQQQQQ", f"hf_***\n***{'Q' * 14}"),
        (f"{HF_TOKEN}{OFF}{OTHER_TOKEN}", "hf_*** hf_***"),
        (f"a {OFF}{HF_TOKEN}{OFF}{BOLD}{OTHER_TOKEN}{OFF} b", "a  hf_***  hf_***  b"),
        (f"{HF_TOKEN}{BOLD}{OTHER_TOKEN[:20]}{OFF}{OTHER_TOKEN[20:]}", "hf_*** hf_***"),
        (f"hf_{'q' * 29}\n{OTHER_TOKEN} b", f"hf_{'q' * 29}\nhf_*** b"),
        (f"hf_{'q' * 17}\n{OTHER_TOKEN}", f"hf_{'q' * 17}\nhf_***"),
        ("hf_dummy\nnext line", "hf_dummy\nnext line"),
    ],
)
def test_mask_covers_a_token_cut_by_colour_codes_or_one_line_break(text, expected):
    masked = mask(text)
    assert masked == expected
    assert masked.count("\n") == text.count("\n")
    assert not any(HF_TOKEN[at : at + 8] in masked for at in range(3, len(HF_TOKEN) - 7))
    assert "y" * 8 not in masked


@pytest.mark.parametrize(
    ("text", "lines"),
    [
        (f"ok\ntoken={HF_TOKEN}\n", [2]),
        (f"ok\n{HF_TOKEN[:12]}{OFF}{HF_TOKEN[12:]}\n", [2]),
        (f"ok\n{HF_TOKEN[:12]}{BOLD}{HF_TOKEN[12:]}\n", [2]),
        (f"ok\nretry {HF_TOKEN[:12]}\n{HF_TOKEN[12:]} done\n", [2]),
        (f"{HF_TOKEN}{OFF}{OTHER_TOKEN}\n\n{HF_TOKEN[:3]}\n{OTHER_TOKEN[3:]}\n", [1, 1, 3]),
        ("see hf_hub\nCollecting driftmap\nhf_dummy\n", []),
    ],
)
def test_scan_flags_a_token_the_way_mask_covers_it(tmp_path, text, lines):
    record = _write(tmp_path / "logs/job.log", text)
    assert scan([record]) == [Finding(record, line, "hf_token") for line in lines]
    assert "hf_***" in mask(text) or not lines


def test_scan_does_not_flag_a_short_run_that_only_continues_into_a_longer_word(tmp_path):
    rows = f"hf_{'q' * 29}\nAKIAQQQQQQQQQQQQQQQ\nwith hf_xet\n{'6f0c65d5' * 8}  model.safetensors\n"
    assert scan([_write(tmp_path / "logs/job.log", rows)]) == []
    cut = f"retry {HF_TOKEN[:12]}\n{HF_TOKEN[12:]} done\n"
    assert [finding.line for finding in scan([_write(tmp_path / "logs/cut.log", cut)])] == [1]


def test_scan_keeps_findings_in_line_order(tmp_path):
    text = f"{HOME_PATHS[0]}/x\n{ACCOUNT} {HF_TOKEN}\n{AWS_KEY}\n{HF_TOKEN[:9]}\n{HF_TOKEN[9:]} {HOME_PATHS[1]}\n"
    record = _write(tmp_path / "runs/r.txt", text)
    hits = [(finding.line, finding.pattern) for finding in scan([record])]
    assert hits == [
        (1, "home_path"),
        (2, "hf_token"),
        (2, "account_id"),
        (3, "aws_key"),
        (4, "hf_token"),
        (5, "home_path"),
    ]


def test_json_escapes_do_not_shield_a_hit(tmp_path):
    record = _write(
        tmp_path / "runs/r.json",
        json.dumps({"error": f"boom\n{HOME_PATHS[0]}/x", "note": f"é{ACCOUNT}\t{ACCOUNT}"}) + "\n",
    )
    assert [finding.pattern for finding in scan([record])] == ["account_id", "account_id", "home_path"]


def test_colour_codes_do_not_shield_a_hit(tmp_path):
    coloured = f"\x1b[2K{HOME_PATHS[0]}/x\x1b[0m \x1b[1m{ACCOUNT}\x1b[0m"
    record = _write(tmp_path / "runs/r.json", json.dumps({"traceback": coloured}) + "\n" + coloured + "\n")
    hits = [(finding.line, finding.pattern) for finding in scan([record])]
    assert hits == [(1, "account_id"), (1, "home_path"), (2, "account_id"), (2, "home_path")]
    for text in (coloured, json.dumps(coloured)):
        assert "alice" not in mask(text)
        assert "<home>/x" in mask(text)


def test_blank_data_root_exempts_nothing(tmp_path, monkeypatch):
    _write(tmp_path / "prompts/en.jsonl", f"{ACCOUNT}\n")
    monkeypatch.chdir(tmp_path)
    assert scan([Path("prompts/en.jsonl")], ".") == []
    assert scan([Path("prompts/en.jsonl")], "") == [Finding(Path("prompts/en.jsonl"), 1, "account_id")]


def test_redaction_finding_names_file_and_line(tmp_path):
    findings = [Finding(tmp_path / "card.md", 3, "hf_token")]
    error = RedactionFinding(findings)
    assert isinstance(error, DriftmapError)
    assert error.exit_code == 6
    assert error.findings == findings
    assert f"{tmp_path / 'card.md'}:3: hf_token" in str(error)


def test_pattern_names():
    assert list(PATTERNS) == ["hf_token", "aws_key", "account_id", "home_path"]
