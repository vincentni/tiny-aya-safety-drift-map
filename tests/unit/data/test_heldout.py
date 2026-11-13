from driftmap.data.heldout import TRAIN_CAP, balance, carve


def rows(language: str, origin: str, count: int) -> list[dict]:
    return [{"prompt_id": f"{language}-{origin}-{i}", "prompt": f"{language} {origin} {i}"} for i in range(count)]


def ids(rows_: list[dict]) -> set[str]:
    return {row["prompt_id"] for row in rows_}


TRAIN_SIZES = {"en": 500, "ar": 480, "fr": 300, "es": 350, "ru": 423, "hi": 250}


def test_heldout_is_carved_by_split_and_disjoint_from_training():
    """100 rows per language, en and ar from test, the rest carved from train, no id shared (TS-DATA-4)."""
    train = {language: rows(language, "train", count) for language, count in TRAIN_SIZES.items()}
    test = {"en": rows("en", "test", 250), "ar": rows("ar", "test", 250), "fr": rows("fr", "test", 40)}
    # every en test prompt also sits in train
    train["en"] += rows("en", "test", 250)

    result = carve(train, test, 100, seed=20261102)
    balanced = balance(result.pools, seed=20261102)

    assert result.origin == {"en": "test", "ar": "test", "fr": "train", "es": "train", "ru": "train", "hi": "train"}
    for language in TRAIN_SIZES:
        held, pool, kept = ids(result.heldout[language]), ids(result.pools[language]), ids(balanced[language])
        assert len(result.heldout[language]) == len(held) == 100
        assert not held & pool and kept <= pool
        source = test if result.origin[language] == "test" else train
        assert held <= ids(source[language])
        assert pool <= ids(train[language])
    assert not ids(sum(result.heldout.values(), [])) & ids(sum(result.pools.values(), []))
    assert result.duplicates == 100
    assert {language: len(pool) for language, pool in result.pools.items()} == {
        "en": 650,
        "ar": 480,
        "fr": 200,
        "es": 250,
        "ru": 323,
        "hi": 150,
    }
    # hi has the smallest pool after its carve
    assert {language: len(kept) for language, kept in balanced.items()} == dict.fromkeys(TRAIN_SIZES, 150)
    assert carve(train, test, 100, seed=20261102) == result
    assert balance(result.pools, seed=20261102) == balanced
    assert carve(train, test, 100, seed=1).heldout != result.heldout
    assert balance(result.pools, seed=1) != balanced


def test_training_is_capped_per_language():
    pools = {language: rows(language, "train", 700) for language in ("en", "ru")}
    balanced = balance(pools, seed=3)
    assert TRAIN_CAP == 423
    assert [len(balanced[language]) for language in ("en", "ru")] == [423, 423]
    assert len(balance(pools, seed=3, cap=50)["ru"]) == 50
    kept = [row["prompt_id"] for row in balanced["en"]]
    assert kept == sorted(kept, key=lambda name: int(name.rsplit("-", 1)[1])) and len(set(kept)) == 423
    assert balance({"en": pools["en"], "ru": []}, seed=3) == {"en": [], "ru": []}


def test_duplicate_prompts_never_reach_both_sides():
    twin = {"prompt_id": "fr-0", "prompt": "fr 0", "target": "other"}
    train = {"fr": [{"prompt_id": "fr-0", "prompt": "fr 0", "target": "one"}, twin]}
    result = carve(train, {}, 1, seed=3)
    assert ids(result.heldout["fr"]) == {"fr-0"} and len(result.heldout["fr"]) == 1
    assert result.pools["fr"] == []
    assert result.duplicates == 1


def test_repeated_prompts_never_fill_two_heldout_slots():
    repeated = [{"prompt_id": "a", "prompt": "a", "target": str(i)} for i in range(10)]
    others = [{"prompt_id": name, "prompt": name, "target": "t"} for name in ("b", "c")]

    for seed in range(5):
        carved = carve({"fr": repeated + others}, {}, 3, seed=seed)
        assert sorted(row["prompt_id"] for row in carved.heldout["fr"]) == ["a", "b", "c"]
        assert carved.heldout["fr"][0] == repeated[0]
        assert carved.pools["fr"] == [] and carved.duplicates == 9

        tested = carve({"en": rows("en", "train", 4)}, {"en": repeated + others}, 3, seed=seed)
        assert tested.origin == {"en": "test"}
        assert sorted(row["prompt_id"] for row in tested.heldout["en"]) == ["a", "b", "c"]
        assert len(tested.pools["en"]) == 4 and tested.duplicates == 0

    # two distinct test prompts are fewer than n, so the rows are carved from train
    assert carve({"en": rows("en", "train", 4)}, {"en": repeated + others[:1]}, 3, seed=0).origin == {"en": "train"}


def test_short_language_keeps_what_it_has():
    result = carve({"hi": rows("hi", "train", 30)}, {}, 100, seed=3)
    assert len(result.heldout["hi"]) == 30
    assert result.pools["hi"] == []
