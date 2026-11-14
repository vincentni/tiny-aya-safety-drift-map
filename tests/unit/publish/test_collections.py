import pytest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.publish import collections
from driftmap.publish.push import publish
from tests.fakes import FakeHub
from tests.fakes.labels import OWNER, make_settings
from tests.fakes.published import DATASETS, MERGED, MODELS, REPOS, SMOKE, seed, writes


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def fake(tmp_path):
    return FakeHub(tmp_path / "hub")


def listed(fake, name):
    record = next(record for record in fake.collections.values() if record["title"] == name)
    return [(item["id"], item["type"]) for item in record["items"]]


def test_each_public_repo_is_listed_once_and_the_merged_one_never(settings, fake):
    """Models and datasets land in their own collection; a second run adds nothing (TS-PUB-5)."""
    hub = seed(fake)

    publish(settings, hub, echo=lambda line: None)

    assert listed(fake, REPOS["models_collection"]) == [(repo, "model") for repo in MODELS]
    assert listed(fake, REPOS["data_collection"]) == [(repo, "dataset") for repo in DATASETS]
    assert not any(method == "create_collection" for method, _ in fake.calls)
    fake.calls.clear()
    publish(settings, hub, echo=lambda line: None)
    assert writes(fake) == []
    everything = listed(fake, REPOS["models_collection"]) + listed(fake, REPOS["data_collection"])
    assert not {MERGED, *SMOKE} & {repo for repo, _ in everything}


def test_a_collection_is_found_by_title_or_slug_with_all_its_items(settings, fake):
    hub = seed(fake)
    fake.collections.pop(next(slug for slug in fake.collections if "models" in slug))
    renamed = f"{OWNER}/{REPOS['models_collection']}-{'0' * 24}"
    fake.collections[renamed] = {
        "slug": renamed,
        "title": "Tiny Aya safety drift map: models",
        "owner": {"name": OWNER, "type": "user"},
        "items": [{"_id": f"i{n}", "id": f"{OWNER}/m{n}", "type": "model", "position": n} for n in range(6)],
        "description": None,
        "private": False,
    }
    fake.create_collection(REPOS["data_collection"], namespace="someone-else")

    found = collections.find(settings, hub)

    assert found["models"].slug == renamed and len(found["models"].items) == 6
    assert found["data"].owner["name"] == OWNER
    assert collections.holds(found["models"], f"{OWNER}/m5", "model")
    assert not collections.holds(found["models"], f"{OWNER}/m5", "dataset")


def test_an_absent_collection_is_none_and_two_matches_refuse(settings, fake):
    hub = seed(fake)
    data = next(slug for slug in fake.collections if "data" in slug)
    record = fake.collections.pop(data)

    assert collections.find(settings, hub)["data"] is None

    for suffix in ("1" * 24, "2" * 24):
        slug = f"{OWNER}/{REPOS['data_collection']}-{suffix}"
        fake.collections[slug] = {**record, "slug": slug, "title": "data"}
    with pytest.raises(DriftmapError, match="2 collections match") as refused:
        collections.find(settings, hub)
    assert refused.value.exit_code == ExitCode.INTEGRITY
