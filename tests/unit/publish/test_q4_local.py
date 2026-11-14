from driftmap.publish.q4_local import ANSWERS, LABELS, LABELS_FULL, META, Count, measured
from tests.fakes.published import GGUF_SHA256, q4_check_files
from tests.fakes.published import q4_label as label


def test_the_fixture_names_the_files_as_the_check_does():
    assert set(q4_check_files({}, {})) == {META, ANSWERS, LABELS, LABELS_FULL}


def test_each_label_is_counted_over_the_pairs_scored_at_both_precisions():
    q4 = {
        "h1": label("yes"),
        "h2": label("no", "yes"),
        "h3": label("yes"),
        # no verdict at 4 bits, or a label the judge left open: the pair is not counted
        "h4": None,
        "h5": label("unsure"),
        "x1": label("no", "yes"),
        "x2": label("yes"),
        # benign rows carry no label
        "b1": None,
    }
    full = q4 | {"h1": label("no", "yes"), "h4": label("yes"), "h5": label("yes"), "x2": label("no")}
    sets = {"x1": "xstest", "x2": "xstest", "b1": "benign"}

    found = measured(q4_check_files(q4, full, sets))

    assert (found.sha256, found.llamacpp) == (GGUF_SHA256, "b11960")
    assert found.harmful == Count(pairs=3, full=1, q4=2, only_full=0, only_q4=1)
    assert found.refusal == Count(pairs=3, full=2, q4=1, only_full=1, only_q4=0)
    assert found.over_refusal == Count(pairs=2, full=1, q4=1, only_full=0, only_q4=0)
    assert found.safe_harmful == Count(pairs=2, full=0, q4=1, only_full=0, only_q4=1)


def test_a_check_that_is_incomplete_or_does_not_parse_measures_nothing():
    files = q4_check_files({"h1": label("yes")}, {"h1": label("no")})

    assert measured(files) is not None
    for name in files:
        assert measured({key: data for key, data in files.items() if key != name}) is None
        assert measured(files | {name: b"{cut"}) is None
    assert measured(files | {META: b'{"gguf": {}}\n'}) is None
    assert measured(files | {LABELS: b'{"label": {"response_harm": "yes", "refusal": "no"}}\n'}) is None
    # no pair scored at both precisions is no measurement
    assert measured(q4_check_files({"h1": None}, {"h1": label("no")})) is None
