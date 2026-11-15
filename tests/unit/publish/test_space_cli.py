from types import SimpleNamespace

from typer.testing import CliRunner

from driftmap import cli
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.publish import space


def test_publish_space_hands_the_checkout_and_the_dry_run_flag_to_the_space_step(monkeypatch, tmp_path):
    """The verb needs no readiness report, prints through the masking echo and exits with the step's code."""
    seen = []
    deps = SimpleNamespace(hub="hub", repo_root=tmp_path)
    settings = SimpleNamespace(publish=SimpleNamespace(locked=False, closed=False))
    monkeypatch.setattr(cli, "_open", lambda verb, **rows: seen.append((verb, rows)) or (settings, deps))

    def step(settings, hub, root, *, dry_run, echo, staged, gate):
        # where the checkout stages files and where a check records its passes
        assert (staged, gate) == (tmp_path / ".run-local" / "staged", tmp_path / ".run-local" / "publish-gate.json")
        seen.append((settings, hub, root, dry_run))
        echo(f"upload: index.html from /{'home'}/someone/checkout")

    monkeypatch.setattr(space, "publish_space", step)
    dry = CliRunner().invoke(cli.app, ["publish", "space", "--dry-run"])

    assert dry.exit_code == 0, dry.output
    assert seen == [("publish space", {"ready": False, "dry_run": True}), (settings, "hub", tmp_path, True)]
    assert "upload: index.html from <home>/checkout" in dry.output

    def refuse(settings, hub, root, *, dry_run, echo, **here):
        raise DriftmapError(ExitCode.INTEGRITY, "space/data.json is missing; run `python space/build_data.py` first")

    monkeypatch.setattr(space, "publish_space", refuse)
    refused = CliRunner().invoke(cli.app, ["publish", "space"])

    assert refused.exit_code == ExitCode.INTEGRITY == 12
    assert "error: space/data.json is missing" in refused.output
    assert seen[-1] == ("publish space", {"ready": False, "dry_run": False})
