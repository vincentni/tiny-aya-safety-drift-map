import copy
import unicodedata

import pytest
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample
from pydantic import ValidationError

from driftmap.data.dedup import THRESHOLD, cross_dedup

PLANTS = (
    "Explain how green plants turn sunlight, water and carbon dioxide into sugar and oxygen "
    "during the long summer days."
)
PLANTS_NEAR = PLANTS.replace("how green plants", "how the green plants")
FESTIVAL = (
    "the quiet river town of zorblat holds a lantern festival every autumn where bakers trade cinnamon bread for "
    "carved wooden boats while children paint blue kites and elders tell long stories about the first winter market "
    "near the old stone bridge"
)
# two words changed: trigram similarity 0.89, yet the library's signatures miss it
FESTIVAL_NEAR = FESTIVAL.replace("the quiet river", "the saffron river").replace(
    "the first winter", "the marble winter"
)
FESTIVAL_FAR = FESTIVAL_NEAR.replace("lantern festival", "harvest parade").replace("blue kites", "red masks")
OTHER = "Name two imaginary colours a child might invent for a drawing of the moon."
LIBRARY = "Où se trouve la bibliothèque municipale près du marché de Noël cette année ?"
STREET = "Die große Straße führt zum alten Schloß hinter dem Fluß."


def row(prompt: str, tag: str = "") -> dict:
    return {"prompt_id": f"{tag}{len(prompt)}", "prompt": prompt, "target": f"answer {tag}"}


def similarity(a: str, b: str) -> float:
    first, second = ({text[i : i + 3] for i in range(len(text) - 2)} for text in (a, b))
    return len(first & second) / len(first | second)


def library_survivors(*texts: str) -> list[str]:
    samples = [DataSample(id=str(i), source_uri="synthetic", instruction=text) for i, text in enumerate(texts)]
    return [sample.id for sample in MinHashDeduplicator(threshold=THRESHOLD).run(ExactDeduplicator().run(samples))]


def test_training_copies_of_heldout_prompts_are_dropped():
    """One exact and one near-duplicate training row are dropped, held-out rows untouched (TS-DATA-7)."""
    heldout = [row(PLANTS), row(FESTIVAL)]
    before = copy.deepcopy(heldout)
    train = [row(FESTIVAL.upper(), "t"), row(OTHER, "t"), row(PLANTS_NEAR, "t")]

    kept, dropped_by_dedup, repeats = cross_dedup(heldout, train)

    assert kept == [train[1]]
    assert dropped_by_dedup == 2
    assert repeats == 0
    assert heldout == before
    assert cross_dedup([], train) == (train, 0, 0)
    assert cross_dedup(heldout, []) == ([], 0, 0)


def test_library_dedup_keeps_the_first_seen_row():
    """Protected rows fed first survive exact and MinHash dedup, the training copies go (TS-DATA-9)."""
    texts = {"heldout-0": FESTIVAL, "heldout-1": PLANTS, "train-0": FESTIVAL, "train-1": PLANTS_NEAR, "train-2": OTHER}
    samples = [DataSample(id=name, source_uri="synthetic", instruction=text) for name, text in texts.items()]

    exact = ExactDeduplicator().run(samples)
    near = MinHashDeduplicator(threshold=0.85).run(exact)

    assert [sample.id for sample in exact] == ["heldout-0", "heldout-1", "train-1", "train-2"]
    assert [sample.id for sample in near] == ["heldout-0", "heldout-1", "train-2"]
    assert len(samples) - len(near) == 2
    assert THRESHOLD == 0.85
    with pytest.raises(ValidationError):
        DataSample(instruction=OTHER)


def test_a_near_copy_the_library_misses_is_still_dropped():
    assert 0.85 <= similarity(FESTIVAL, FESTIVAL_NEAR) < 0.90
    assert library_survivors(FESTIVAL, FESTIVAL_NEAR) == ["0", "1"]
    assert similarity(FESTIVAL, FESTIVAL_FAR) < 0.85

    train = [row(FESTIVAL_NEAR, "t"), row(FESTIVAL_FAR, "t"), row(OTHER, "t")]
    kept, copies, repeats = cross_dedup([row(PLANTS), row(FESTIVAL)], train)

    assert kept == train[1:]
    assert (copies, repeats) == (1, 0)


@pytest.mark.parametrize(
    "protected, copy_",
    [
        (unicodedata.normalize("NFC", LIBRARY), unicodedata.normalize("NFD", LIBRARY)),
        (STREET, STREET.replace("ß", "ss").upper()),
        (
            "Count the stars: １２３ ｓｔａｒｓ in the tiny synthetic sky tonight.",
            "count the stars: 123 stars in the tiny synthetic sky tonight.",
        ),
    ],
)
def test_another_unicode_form_of_a_prompt_is_a_copy(protected, copy_):
    assert protected != copy_ and library_survivors(protected, copy_) == ["0", "1"]
    assert cross_dedup([row(protected)], [row(copy_, "t"), row(OTHER, "t")]) == ([row(OTHER, "t")], 1, 0)


def test_training_only_repeats_are_counted_apart():
    train = [row(OTHER, "a"), row(PLANTS, "a"), row(OTHER.upper(), "b"), row(PLANTS_NEAR, "b"), row(STREET, "a")]

    assert cross_dedup([row(FESTIVAL)], train) == ([train[0], train[1], train[4]], 0, 2)
    assert cross_dedup([], train) == ([train[0], train[1], train[4]], 0, 2)
    # a row that is both counts as a copy of the protected prompt
    assert cross_dedup([row(PLANTS)], train) == ([train[0], train[4]], 2, 1)


def test_only_the_prompt_is_compared():
    train = [{"prompt_id": "t", "prompt": OTHER, "target": "a different completion"}]
    assert cross_dedup([{"prompt_id": "h", "prompt": OTHER}], train) == ([], 1, 0)


def test_short_prompts_never_match_on_empty_trigram_sets():
    kept, copies, repeats = cross_dedup([row(FESTIVAL), row("")], [row(OTHER, "t")])
    assert (kept, copies, repeats) == ([row(OTHER, "t")], 0, 0)


def test_a_copy_only_the_library_sees_is_dropped():
    spaced = OTHER.replace(" ", "\n\n")
    assert similarity(OTHER.casefold(), spaced.casefold()) < 0.85
    assert cross_dedup([row(OTHER)], [row(spaced, "t"), row(PLANTS, "t")]) == ([row(PLANTS, "t")], 1, 0)
