import json
import re
import tomllib
from datetime import date
from pathlib import Path

import huggingface_hub
import pytest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import HASHED, JUDGE_LANGUAGES, ConfigError, Settings, load_settings
from tests.fakes.settings import BASE, settings_toml, to_toml

REPO_ROOT = Path(__file__).resolve().parents[3]
PLANTED_TOKEN = "hf_" + "x" * 34


def _reversed(data):
    if isinstance(data, dict):
        return {key: _reversed(value) for key, value in reversed(data.items())}
    return data


def _keys(data, prefix=""):
    for key, value in data.items():
        yield prefix + key
        if isinstance(value, dict) and key not in ("revisions", "prices_micro_usd_per_min"):
            yield from _keys(value, f"{prefix}{key}.")


def test_full_toml_loads_and_a_planted_token_key_is_refused(tmp_path, monkeypatch):
    """A secret key in the file is refused before any client exists (TS-CFG-1)."""
    built = []
    monkeypatch.setattr(huggingface_hub.HfApi, "__init__", lambda self, *a, **k: built.append(self))

    settings = load_settings(settings_toml(tmp_path))
    assert isinstance(settings, Settings)
    assert settings.owner == "vincentnkq"
    assert settings.lock_path == tmp_path / "launch.lock"

    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, token=PLANTED_TOKEN))
    assert caught.value.exit_code is ExitCode.USAGE
    assert isinstance(caught.value, DriftmapError)
    assert str(caught.value) == "token: Extra inputs are not permitted [extra_forbidden]"
    assert PLANTED_TOKEN not in repr(caught.value)
    assert built == []


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"bogus": 1}, "bogus"),
        ({"languages": None}, "languages"),
        ({"ceilings": {"hard": "20000000"}}, "ceilings.hard"),
        ({"llamacpp_release": {"sha256": None}}, "llamacpp_release.sha256"),
    ],
    ids=["unknown-key", "missing-languages", "string-ceiling", "missing-sha256"],
)
def test_malformed_toml_names_the_field_path(tmp_path, overrides, field):
    """Each malformed file exits 2 with the field path first (TS-CFG-2)."""
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, **overrides))
    assert caught.value.exit_code == 2
    lines = str(caught.value).splitlines()
    assert len(lines) == 1
    assert lines[0].split(": ")[0] == field


def test_language_outside_the_judge_set_is_refused(tmp_path):
    """A language the judge cannot score is refused as judge_unsupported (TS-CFG-3)."""
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, languages=["en", "sr"]))
    assert caught.value.exit_code == 2
    assert str(caught.value) == "languages.1: sr is not a PolyGuard language [judge_unsupported]"

    assert load_settings(settings_toml(tmp_path)).languages == ["en", "fr", "es", "ru", "hi", "ar"]
    assert len(JUDGE_LANGUAGES) == 17
    assert load_settings(settings_toml(tmp_path, languages=sorted(JUDGE_LANGUAGES))).languages == sorted(
        JUDGE_LANGUAGES
    )


def test_config_hash_ignores_key_order_and_moves_with_parser_and_batch(tmp_path):
    """Key order leaves the hash alone; parser version and an OOM rung move it (TS-CFG-4)."""
    first = load_settings(settings_toml(tmp_path))

    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    data = tomllib.loads((tmp_path / "driftmap.toml").read_text(encoding="utf-8"))
    (shuffled / "driftmap.toml").write_text(to_toml(_reversed(data)), encoding="utf-8")
    (shuffled / "revisions.json").write_text(json.dumps(_reversed(BASE["revisions"])), encoding="utf-8")
    second = load_settings(shuffled / "driftmap.toml")
    assert list(second.revisions) == list(reversed(first.revisions))

    parser = load_settings(settings_toml(tmp_path, judge={"parser_version": "2"}))
    rung = load_settings(settings_toml(tmp_path, batch={"generate": 8}))

    assert re.fullmatch(r"[0-9a-f]{64}", first.config_hash())
    assert first.config_hash() == second.config_hash()
    assert len({first.config_hash(), parser.config_hash(), rung.config_hash()}) == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"ceilings": {"own_money": 2_000_000}},
        {"timeouts": {"drift": 240}},
        {"publish": {"space_not_serving": True}},
        {"readiness": {"registration_confirmed": False}},
        {"prices_micro_usd_per_min": {"l4x1": 14_000}},
        {"flavors": {"gpu": "a10g-small"}},
        {"runner": "uv"},
        {"scheduling_allowance_s": 600},
        {"code": {"branch": "release"}},
        {"owner": "someone-else"},
    ],
)
def test_operational_edits_keep_the_config_hash(tmp_path, overrides):
    before = load_settings(settings_toml(tmp_path)).config_hash()
    assert load_settings(settings_toml(tmp_path, **overrides)).config_hash() == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"languages": ["en", "fr"]},
        {"n_harmful": 300},
        {"n_heldout": 300},
        {"seed": 1},
        {"sources": {"benign": "CohereLabs/aya_collection"}},
        {"revisions": {"Paul/XSTest": "6" * 40}},
        {"batch": {"drift": 2, "drift_accum": 8}},
        {"decoding": {"max_new_tokens": 128}},
        {"judge": {"dtype": "float32"}},
        {"repair": {"alpha": 0.4}},
        {"repair": {"bars": {"safety_points": 3.0}}},
    ],
)
def test_result_defining_edits_move_the_config_hash(tmp_path, overrides):
    before = load_settings(settings_toml(tmp_path)).config_hash()
    assert load_settings(settings_toml(tmp_path, **overrides)).config_hash() != before
    assert set(overrides) < HASHED
    # preamble has one legal value, so only its membership can be checked
    assert "preamble" in HASHED
    assert len(HASHED) == 11


def test_lock_path_is_expanded_and_stays_out_of_the_hash(tmp_path):
    home = load_settings(settings_toml(tmp_path, lock_path="~/.driftmap/launch.lock"))
    assert home.lock_path == Path.home() / ".driftmap" / "launch.lock"
    assert tmp_path in home.lock_path.parents
    assert home.config_hash() == load_settings(settings_toml(tmp_path)).config_hash()


def test_default_lock_path_is_expanded(tmp_path):
    path = settings_toml(tmp_path, lock_path=None)
    assert "lock_path" not in tomllib.loads(path.read_text(encoding="utf-8"))
    assert load_settings(path).lock_path == Path.home() / ".driftmap" / "launch.lock"


def test_revisions_come_from_the_sibling_json(tmp_path):
    pinned = load_settings(settings_toml(tmp_path))
    assert pinned.revisions == BASE["revisions"]

    bare = tmp_path / "bare"
    bare.mkdir()
    unpinned = load_settings(settings_toml(bare, revisions=None))
    assert not (bare / "revisions.json").exists()
    assert unpinned.revisions == {}
    assert unpinned.config_hash() != pinned.config_hash()


def test_a_branch_name_as_revision_is_refused(tmp_path):
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, revisions={"CohereLabs/tiny-aya-global": "main"}))
    assert str(caught.value).startswith("revisions.CohereLabs/tiny-aya-global: ")


@pytest.mark.parametrize(
    ("name", "text"),
    [("driftmap.toml", "owner = "), ("revisions.json", "{")],
    ids=["toml", "revisions"],
)
def test_unparsable_files_exit_2_naming_the_file(tmp_path, name, text):
    path = settings_toml(tmp_path)
    (tmp_path / name).write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_settings(path)
    assert caught.value.exit_code == 2
    assert str(caught.value).startswith(f"{tmp_path / name}: ")


def test_missing_file_exits_2(tmp_path):
    with pytest.raises(ConfigError) as caught:
        load_settings(tmp_path / "driftmap.toml")
    assert caught.value.exit_code == 2
    assert "driftmap.toml" in str(caught.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeouts": {"drift": 0}},
        {"timeouts": {"drift": "180"}},
        {"batch": {"judge": 0}},
        {"n_harmful": -1},
        {"prices_micro_usd_per_min": {"l4x1": 13333.0}},
        {"preamble": "strip"},
        {"runner": "docker"},
        {"judge": {"revision": "main"}},
        {"judge": {"smol_revision": "main"}},
        {"llamacpp_release": {"sha256": {"cpu_asset": "31cc3a6b"}}},
        {"repair": {"alphas": [0.1, 0.2, 0.3, 0.4, 0.5]}},
        {"repair": {"alphas": []}},
        {"images": {"llamacpp_full": "ghcr.io/ggml-org/llama.cpp:full"}},
    ],
)
def test_out_of_range_values_are_refused(tmp_path, overrides):
    with pytest.raises(ConfigError):
        load_settings(settings_toml(tmp_path, **overrides))


@pytest.mark.parametrize(
    ("overrides", "start"),
    [
        ({"prices_micro_usd_per_min": {"l4x1": 0}}, "prices_micro_usd_per_min.l4x1: "),
        ({"prices_micro_usd_per_min": {"cpu-basic": -167}}, "prices_micro_usd_per_min.cpu-basic: "),
        ({"prices_micro_usd_per_min": {"l4x1": None}}, "flavors: Value error, no price for l4x1"),
        ({"flavors": {"gpu": "t4-small"}}, "flavors: Value error, no price for t4-small"),
        ({"flavors": {"cpu": "cpu-xl"}}, "flavors: Value error, no price for cpu-xl"),
        ({"ceilings": {"own_money": -1}}, "ceilings.own_money: "),
        ({"ceilings": {"envelope": -1}}, "ceilings.envelope: "),
        ({"ceilings": {"smoke": 5_000_001}}, "ceilings: Value error, expected smoke <= per_run <= soft <= hard"),
        ({"ceilings": {"per_run": 16_000_001}}, "ceilings: Value error, expected smoke <= per_run <= soft <= hard"),
        ({"ceilings": {"soft": 20_000_001}}, "ceilings: Value error, expected smoke <= per_run <= soft <= hard"),
    ],
)
def test_prices_and_ceilings_must_be_consistent(tmp_path, overrides, start):
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, **overrides))
    assert str(caught.value).startswith(start)
    assert len(str(caught.value).splitlines()) == 1


def test_equal_or_zero_ceilings_load(tmp_path):
    equal = load_settings(
        settings_toml(tmp_path, ceilings={"smoke": 5_000_000, "per_run": 5_000_000, "soft": 5_000_000})
    )
    assert (equal.ceilings.smoke, equal.ceilings.soft) == (5_000_000, 5_000_000)
    zero = load_settings(
        settings_toml(tmp_path, ceilings={"smoke": 0, "per_run": 0, "soft": 0, "hard": 0, "envelope": 0})
    )
    assert zero.ceilings.hard == 0


@pytest.mark.parametrize(
    "origin",
    [
        "http://github.com/vincentni/tiny-aya-safety-drift-map",
        "https://gitlab.com/vincentni/tiny-aya-safety-drift-map",
        "https://github.com/vincentni",
        "https://github.com/vincentni/",
        "https://github.com/vincentni/tiny-aya-safety-drift-map/",
        "https://github.com/vincentni/tiny-aya-safety-drift-map/tree/main",
        "https://github.com/vincentni/tiny-aya-safety-drift-map.git",
        "https://github.com/vincentni/..",
        "https://github.com/vincentni/repo' ; id #",
        "https://github.com/vincentni/repo$(id)",
        "https://github.com/vincentni/repo\n",
        "https://github.com.evil.example/vincentni/repo",
        "https://githubxcom/vincentni/repo",
        "github.com:vincentni/tiny-aya-safety-drift-map",
    ],
)
def test_code_origin_must_be_a_github_repo_url(tmp_path, origin):
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, code={"origin": origin}))
    assert str(caught.value).startswith("code.origin: ")


@pytest.mark.parametrize("branch", ["-main", "a..b", "a b", "", "main;id", "main\n", "$(id)", "feature@{1}"])
def test_code_branch_must_be_a_plain_ref_name(tmp_path, branch):
    with pytest.raises(ConfigError) as caught:
        load_settings(settings_toml(tmp_path, code={"branch": branch}))
    assert str(caught.value).startswith("code.branch: ")


def test_plain_origins_and_branches_load(tmp_path):
    code = load_settings(
        settings_toml(tmp_path, code={"origin": "https://github.com/a-b_c/d.e-f", "branch": "release/1.0-rc_1"})
    ).code
    assert (code.origin, code.branch) == ("https://github.com/a-b_c/d.e-f", "release/1.0-rc_1")


def test_timeouts_convert_to_seconds(tmp_path):
    timeouts = load_settings(settings_toml(tmp_path)).timeouts
    seconds = {kind: timeouts.seconds(kind) for kind in type(timeouts).model_fields}
    assert seconds == {
        "drift": 10_800,
        "generate": 5400,
        "judge": 2700,
        "repair": 7200,
        "steer": 5400,
        "quant": 3600,
        "quant_eval": 7200,
        "smoke": 1800,
    }
    assert all(type(value) is int for value in seconds.values())
    with pytest.raises(KeyError):
        timeouts.seconds("train")


def test_repo_toml_loads_with_the_agreed_values():
    settings = load_settings(REPO_ROOT / "driftmap.toml")

    assert settings.owner == "vincentnkq"
    assert settings.languages == ["en", "fr", "es", "ru", "hi", "ar"]
    assert (settings.n_harmful, settings.n_heldout) == (600, 600)
    assert settings.repos.model_dump() == BASE["repos"]
    assert settings.repos.repaired == "tiny-aya-global-repaired"
    assert settings.sources.model_dump() == BASE["sources"]
    assert settings.gated_repos == ["CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"]
    # drift is re-priced after the smoke run
    assert {**settings.timeouts.model_dump(), "drift": 180} == BASE["timeouts"]
    gpu_price = settings.prices_micro_usd_per_min[settings.flavors.gpu]
    for kind, minutes in settings.timeouts.model_dump().items():
        assert minutes * gpu_price <= settings.ceilings.per_run, kind
    assert settings.llamacpp_release.tag == "b11960"
    assert settings.llamacpp_release.model_dump() == BASE["llamacpp_release"]
    assert settings.images.model_dump() == {"pytorch": "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"}
    assert settings.prices_micro_usd_per_min == BASE["prices_micro_usd_per_min"]
    assert settings.judge.model_dump() == BASE["judge"]
    assert settings.decoding.max_new_tokens == 256
    assert (settings.runner, settings.preamble, settings.scheduling_allowance_s) == ("pip", "keep", 1800)
    assert settings.lock_path == Path.home() / ".driftmap" / "launch.lock"
    assert settings.code.model_dump() == BASE["code"]
    assert settings.repair.model_dump() == BASE["repair"]
    assert settings.readiness.registration_confirmed is True
    assert settings.readiness.confirmed_on == date(2026, 11, 7)
    assert isinstance(settings.publish.space_not_serving, bool)
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in settings.revisions.values())


def test_repo_toml_and_the_fake_have_the_same_keys():
    real = tomllib.loads((REPO_ROOT / "driftmap.toml").read_text(encoding="utf-8"))
    fake = {**BASE, "lock_path": ""}
    del fake["revisions"]
    assert sorted(_keys(real)) == sorted(_keys(fake))


def test_repo_config_files_hold_no_secret_or_home_path():
    assert isinstance(json.loads((REPO_ROOT / "revisions.json").read_text(encoding="utf-8")), dict)
    secret = re.compile(r"hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|\b\d{12}\b|(?<![\w/])/(home|Users|local/home)/")
    for name in ("driftmap.toml", "revisions.json"):
        assert not secret.search((REPO_ROOT / name).read_text(encoding="utf-8")), name
