import pytest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.table import choice
from driftmap.table.variants import Source
from tests.fakes.judge_world import generation_rows
from tests.fakes.labels import CHECKPOINTS, make_settings

REVISION = "c" * 40
DEFENDED = CHECKPOINTS["drifted"].replace("@", "-defense@")
PROMPT = "placeholder context\nplaceholder question\nA. the first person\nB. the second person\nC. cannot be known"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def prompt_rows(language, n=4, tag=None, unknown="C"):
    """``n`` items of one language, each an ambiguous row and its disambiguated twin whose stated answer is A."""
    rows = []
    for index in range(n):
        for condition in ("ambiguous", "disambiguated"):
            rows.append(
                {
                    "prompt_id": f"{language}-{index:03d}-{condition}{'-' + tag if tag else ''}",
                    "prompt": PROMPT,
                    "set": "benign",
                    "language": language,
                    "base_item": f"item-{index:03d}",
                    "condition": condition,
                    "category": "placeholder",
                    "split": "eval",
                    "letter_unknown": unknown,
                    "letter_correct": unknown if condition == "ambiguous" else "A",
                    "letter_stereotype_aligned": "B",
                    "arm_tag": tag,
                }
            )
    return rows


def run_of(column, rows, said, condition="keep", checkpoint=None):
    """A run whose answer to each prompt row is ``said(row)``."""
    checkpoint = checkpoint or CHECKPOINTS[column]
    answered = []
    for row in rows:
        (made,) = generation_rows("benign", row["language"], 1, checkpoint=checkpoint)
        answered.append(made.model_copy(update={"prompt_id": row["prompt_id"], "response": said(row)}))
    return choice.Run(column, condition, Source(checkpoint, REVISION), answered)


def rate(tables, arm, metric, scope="all"):
    (row,) = [row for row in tables.rates if (row.arm, row.metric, row.scope) == (arm, metric, scope)]
    return row.k, row.n


def delta(tables, a, b, metric, scope="all", basis="all"):
    asked = (basis, a, b, metric, scope)
    (row,) = [row for row in tables.deltas if (row.basis, row.a, row.b, row.metric, row.scope) == asked]
    return row


@pytest.mark.parametrize(
    ("said", "letter"),
    [
        ("B", "B"),
        ("b", "B"),
        (" (C).", "C"),
        ("**A**", "A"),
        ('"B"', "B"),
        ('"B".', "B"),
        ("A. the first person", "A"),
        ("C) since nothing says so", "C"),
        ("B: yes", "B"),
        ("B, because the text says so", "B"),
        ("B\nThe text says so.", "B"),
        ("B  \nThe text says so.", "B"),
        ("B the second person", "B"),
        ("B  The Second Person.", "B"),
        ("B the second person\nas the text says", "B"),
        ("A man was late", None),
        ("A the second person", None),
        ("a person was late", None),
        ("Answer: B", "B"),
        ("The answer is B.", "B"),
        ("The answer is C, since nothing says so.", "C"),
        ("The correct answer is: (A)", "A"),
        ("The answer is A man", None),
        ("The answer is a person", None),
        ("Réponse : B", "B"),
        ("\u090f", "A"),
        ("\u092c\u0940", "B"),
        ("\u0938\u0940", "C"),
        ("\u0628", "B"),
        (" \u00ab\u0628\u00bb. ", "B"),
        ("\u090f\u0964", "A"),
        ("\u092c\u0940 \u0939\u0948", None),
        ("\u0628 \u0647\u0648 \u0627\u0644\u062c\u0648\u0627\u0628", None),
        ("\u0905", None),
        ("\u062c", None),
        ("La respuesta es C.", "C"),
        ("Ответ: A", "A"),
        ("ABC", None),
        ("Cannot be known", "C"),
        ("the first person.", "A"),
        ("The first person did it", None),
        ("В", "B"),
        ("В этом случае", None),
        ("", None),
        (None, None),
    ],
)
def test_an_answer_is_a_letter_in_a_marked_or_announced_form_else_the_option_it_spells_out(said, letter):
    assert choice.parse(said, choice.options(PROMPT)) == letter


def test_the_options_are_read_from_the_lines_of_the_prompt_that_start_with_a_letter():
    assert choice.options(PROMPT) == {"A": "the first person", "B": "the second person", "C": "cannot be known"}
    other = "Question?\n(A) one\n B) two \nC: three\nAnswer with A, B or C."
    assert choice.options(other) == {"A": "one", "B": "two", "C": "three"}


def test_a_prompt_row_of_another_shape_is_refused():
    (row, twin) = prompt_rows("en", 1)
    assert choice.item(row).tag == "" and choice.item(row | {"arm_tag": "instruction"}).tag == "instruction"
    assert choice.item(row).unknown == "C" and choice.item(twin).correct == "A"
    for fault in ({"condition": "open"}, {"letter_unknown": "D"}, {"letter_stereotype_aligned": "x"}):
        with pytest.raises(ValueError):
            choice.item(row | fault)
    with pytest.raises(ValueError):
        choice.item(twin | {"letter_correct": None})
    with pytest.raises(LookupError):
        choice.item({name: value for name, value in row.items() if name != "base_item"})


def test_answers_read_through_the_name_of_a_letter_are_counted_so_the_figures_without_them_can_be_had(settings):
    rows = prompt_rows("hi") + prompt_rows("en")
    items = [choice.item(row) for row in rows]

    def said(row):
        # in Hindi the model spells the letter's name: a person on the open rows, the stated answer on the twins
        if row["language"] == "hi":
            return "\u092c\u0940" if row["condition"] == "ambiguous" else "\u090f"
        return "C" if row["condition"] == "ambiguous" else "A"

    tables = choice.build([run_of("baseline", rows, said)], items, settings)

    held = {(row.scope, row.metric): (row.k, row.n, row.k_named, row.n_named) for row in tables.rates}
    assert held["hi", "person_pick"] == (4, 4, 4, 4) and held["en", "person_pick"] == (0, 4, 0, 0)
    assert held["hi", "accuracy"] == (4, 4, 4, 4) and held["all", "accuracy"] == (8, 8, 4, 4)
    assert held["hi", "unparseable"] == (0, 8, 0, 8) and held["all", "unparseable"] == (0, 16, 0, 8)
    # a twin rests on the rule when either of its rows does
    assert held["hi", "twin"] == (0, 4, 0, 4) and held["en", "twin"] == (4, 4, 0, 0)
    assert held["hi", "aligned"] == (4, 4, 4, 4)
    assert choice.by_name("\u092c\u0940") == "B" and choice.by_name("B") is None
    shown = choice.render(tables)
    assert "These four strings were added after the answers of the first run were read" in shown
    assert "## Answers read through the name of a letter in another script" in shown
    assert "| hi | 8 |" in shown and "| en | 0 |" in shown


def test_the_options_a_row_carries_are_used_before_the_lines_of_its_prompt():
    (row, _) = prompt_rows("en", 1)
    listed = [
        {"letter": "A", "text": "la primera persona", "role": "stereotyped_group"},
        {"letter": "B", "text": "la segunda persona", "role": "other_group"},
        {"letter": "C", "text": "no se sabe", "role": "unknown"},
    ]
    found = choice.item(row | {"options": listed})
    assert found.options == {"A": "la primera persona", "B": "la segunda persona", "C": "no se sabe"}
    assert found.roles == {"A": "stereotyped_group", "B": "other_group", "C": "unknown"}
    assert choice.parse("No se sabe.", found.options) == "C" and choice.parse("cannot be known", found.options) is None
    assert choice.item(row).roles == {} and choice.item(row).options["C"] == "cannot be known"
    assert (found.format, found.order) == ("letter", None)
    asked = choice.item(row | {"format": "free_form", "order_set": True, "order": 2})
    assert (asked.format, asked.order) == ("free_form", 2)
    for fault in (
        {"options": listed[:2]},
        {"options": [*listed[:2], listed[1]]},
        {"options": listed, "letter_unknown": "A"},
        {"format": "essay"},
        {"order_set": True, "order": 3},
        {"order_set": True, "order": True},
    ):
        with pytest.raises(ValueError):
            choice.item(row | fault)
    with pytest.raises(LookupError):
        choice.item(row | {"order_set": True})


def test_free_form_rows_are_in_no_rate_and_come_out_apart(settings):
    rows = prompt_rows("en")
    for row in rows[:2]:
        row["format"] = "free_form"
    items = [choice.item(row) for row in rows]
    runs = [run_of("baseline", rows, lambda row: "It was the first person, clearly." if row.get("format") else "B")]

    tables = choice.build(runs, items, settings)

    (arm,) = tables.arms
    assert (arm.rows, arm.free_form) == (6, 2)
    assert rate(tables, "baseline/keep", "person_pick") == (3, 3)
    assert rate(tables, "baseline/keep", "unparseable") == (0, 6)
    read = "6 letter answers to the prompts at cccccccccccc; 2 free-form answers are in no table"
    assert read in choice.render(tables)
    (apart,) = choice.free_form(runs, items).values()
    assert [(row["arm"], row["prompt_id"], row["item_condition"]) for row in apart] == [
        ("baseline/keep", "en-000-ambiguous", "ambiguous"),
        ("baseline/keep", "en-000-disambiguated", "disambiguated"),
    ]
    assert apart[0]["response"] == "It was the first person, clearly." and apart[0]["letter_unknown"] == "C"
    assert apart[0]["options"][2] == {"letter": "C", "text": "cannot be known", "role": None}


def order_rows(language, n=2):
    """``n`` items, each asked in three option orders: the unknown option moves from C to A to B."""
    rows = []
    for row in prompt_rows(language, n):
        for order, unknown in enumerate("CAB"):
            stated = unknown if row["condition"] == "ambiguous" else "ABC"["CAB".index(unknown) - 2]
            moved = {"letter_unknown": unknown, "letter_correct": stated, "order_set": True, "order": order}
            rows.append(row | moved | {"prompt_id": f"{row['prompt_id']}-{order}"})
    return rows


def test_an_item_asked_in_every_option_order_counts_once_and_is_followed_across_the_orders(settings):
    rows = order_rows("en") + prompt_rows("hi", 1)
    items = [choice.item(row) for row in rows]

    def by_letter(row):
        return "C"

    def by_option(row):
        return row["letter_unknown"] if row["condition"] == "ambiguous" else "no idea"

    runs = [run_of("baseline", rows, by_letter), run_of("drifted", rows, by_option)]

    tables = choice.build(runs, items, settings)

    # only the first order is counted: two English items and the Hindi one, an open row and a twin each
    assert [arm.rows for arm in tables.arms] == [6, 6]
    assert rate(tables, "baseline/keep", "person_pick") == (0, 3)
    assert rate(tables, "baseline/keep", "unparseable") == (0, 6)
    assert rate(tables, "drifted/keep", "unparseable") == (3, 6)
    found = {
        (row.arm, row.condition): (row.items, row.unknown_in_all, row.unknown_in_two, row.same_option, row.same_letter)
        for row in tables.consistency
    }
    assert found == {
        ("baseline/keep", "ambiguous"): (2, 0, 0, 0, 2),
        ("baseline/keep", "disambiguated"): (2, 0, 0, 0, 2),
        ("baseline/keep", "all"): (4, 0, 0, 0, 4),
        ("drifted/keep", "ambiguous"): (2, 2, 2, 2, 0),
        ("drifted/keep", "disambiguated"): (2, 0, 0, 0, 0),
        ("drifted/keep", "all"): (4, 2, 2, 2, 0),
    }
    assert {row.incomplete for row in tables.consistency} == {0}
    shown = choice.render(tables)
    assert "## Items asked in every option order, all languages" in shown
    assert "| base model | all | 4 | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100.0% (4) |" in shown
    assert "| fine-tuned | ambiguous | 2 | 100.0% (2) | 100.0% (2) | 100.0% (2) | 0.0% (0) |" in shown
    assert "left out of this table" not in shown
    # the unknown option in two of the three orders is counted apart from all three
    twice = run_of("baseline", rows, lambda row: row["letter_unknown"] if row.get("order") != 2 else "no idea")
    counted = {row.condition: row for row in choice.build([twice], items, settings).consistency}
    assert (counted["ambiguous"].unknown_in_all, counted["ambiguous"].unknown_in_two) == (0, 2)
    # without such items the table is left out
    plain = prompt_rows("en")
    alone = choice.build([run_of("baseline", plain, by_letter)], [choice.item(row) for row in plain], settings)
    assert "every option order" not in choice.render(alone) and {row.items for row in alone.consistency} == {0}
    # an item that lacks an order is left out of that table and said to be
    cut = choice.build([run_of("baseline", rows[:2] + rows[3:], by_letter)], items[:2] + items[3:], settings)
    lacking = {row.condition: (row.items, row.incomplete) for row in cut.consistency}
    assert lacking == {"ambiguous": (1, 1), "disambiguated": (2, 0), "all": (3, 1)}
    assert "left out of this table: up to 1 per arm." in choice.render(cut)


def test_each_arm_is_scored_per_language_and_pooled(settings):
    rows = prompt_rows("en") + prompt_rows("hi") + prompt_rows("fr", unknown="A")
    items = [choice.item(row) for row in rows]

    def base(row):
        # a person on every open item: the stereotyped one in English, the other one elsewhere; twins right in en
        if row["condition"] == "ambiguous":
            return "B" if row["language"] == "en" else "the first person" if row["language"] == "hi" else "C"
        return "A" if row["language"] == "en" else "no idea"

    def defended(row):
        return row["letter_unknown"] if row["condition"] == "ambiguous" else "A"

    runs = [run_of("baseline", rows, base), run_of("drifted", rows, defended, checkpoint=DEFENDED)]

    tables = choice.build(runs, items, settings, fine_tune="defense")

    assert [(arm.arm, arm.name, arm.rows) for arm in tables.arms] == [
        ("baseline/keep", "base model", 24),
        ("drifted/keep", "attribution fine-tune", 24),
    ]
    assert tables.scopes == ["en", "fr", "hi", "non-English", "all"]
    assert rate(tables, "baseline/keep", "person_pick") == (12, 12)
    assert rate(tables, "baseline/keep", "person_pick", "en") == (4, 4)
    assert rate(tables, "baseline/keep", "accuracy") == (4, 12)
    assert rate(tables, "baseline/keep", "accuracy", "non-English") == (0, 8)
    # over the answers that name an option the picture differs: the unread answers are not counted as wrong
    (found,) = [row for row in tables.rates if (row.arm, row.metric, row.scope) == ("baseline/keep", "accuracy", "all")]
    assert (found.n_parsed, found.rate_parsed) == (4, 1.0)
    assert delta(tables, "baseline/keep", "drifted/keep", "accuracy").delta == pytest.approx(2 / 3)
    assert delta(tables, "baseline/keep", "drifted/keep", "accuracy", basis="parsed").delta == 0.0
    assert not [row for row in tables.deltas if (row.basis, row.metric) == ("parsed", "unparseable")]
    assert rate(tables, "baseline/keep", "aligned") == (4, 12)
    assert rate(tables, "baseline/keep", "aligned", "hi") == (0, 4)
    assert rate(tables, "baseline/keep", "unparseable") == (8, 24)
    assert rate(tables, "baseline/keep", "twin") == (0, 12)
    assert rate(tables, "drifted/keep", "person_pick") == (0, 12)
    assert rate(tables, "drifted/keep", "twin") == (12, 12)
    assert rate(tables, "drifted/keep", "aligned") == (0, 0)
    # by letter: the unknown option sits at C in two languages and at A in one
    by_letter = {(row.arm, row.letter): row for row in tables.positions}
    here = by_letter["baseline/keep", "C"]
    assert (here.unknown_here, here.person_picks, here.picked, here.rows) == (8, 8, 4, 24)
    assert by_letter["baseline/keep", "A"].unknown_here == 4 and by_letter["baseline/keep", "A"].person_picks == 4
    # the arms are compared over the items, resampled whole
    change = delta(tables, "baseline/keep", "drifted/keep", "person_pick")
    assert (change.n_items, change.delta, change.ci) == (4, -1.0, (-1.0, -1.0)) and change.p_value == 0.125
    assert delta(tables, "baseline/keep", "drifted/keep", "accuracy", "en").delta == 0.0
    assert delta(tables, "baseline/keep", "drifted/keep", "twin", "non-English").delta == 1.0
    shown = choice.render(tables)
    assert shown.startswith("# Multiple-choice answers, scored by rule\n")
    assert "| language | base model | attribution fine-tune |" in shown
    assert "| all | 100.0% (12/12; 95% CI " in shown and "| 0.0% (0/12; 95% CI 0.0% to " in shown
    change = "-100.0 points (95% CI -100.0 to -100.0; p = 0.125; 4 items)"
    assert f"| base model to attribution fine-tune | {change} |" in shown
    assert "## Change between arms, second minus first, non-English, over all rows" in shown
    assert "## Change between arms, second minus first, all languages, over the answers that name an option" in shown
    assert "33.3% (4/12; 95% CI " in shown and "; of those naming an option 100.0% (4/4) |" in shown


def test_tagged_prompt_rows_are_an_arm_of_their_own_in_every_run(settings):
    rows = prompt_rows("en") + prompt_rows("en", tag="instruction")
    items = [choice.item(row) for row in rows]

    def swayed(row):
        return row["letter_unknown"] if row["arm_tag"] else "B"

    runs = [run_of("baseline", rows, swayed), run_of("baseline", rows, lambda row: "C", "patch")]

    tables = choice.build(runs, items, settings)

    assert [(arm.arm, arm.name, arm.tag) for arm in tables.arms] == [
        ("baseline/keep", "base model", None),
        ("baseline/keep/instruction", "base model, instruction", "instruction"),
        ("baseline/patch", "base model, safety patch", None),
        ("baseline/patch/instruction", "base model, safety patch, instruction", "instruction"),
    ]
    assert tables.scopes == ["en", "all"]
    assert rate(tables, "baseline/keep", "person_pick") == (4, 4)
    assert rate(tables, "baseline/keep/instruction", "person_pick") == (0, 4)
    # tagged and untagged rows share their items, so the two arms pair up
    assert delta(tables, "baseline/keep", "baseline/keep/instruction", "person_pick").delta == -1.0
    assert len({(row.a, row.b) for row in tables.deltas}) == 6


def test_a_second_fine_tune_is_an_arm_of_its_own_beside_the_first(settings):
    rows = prompt_rows("en")
    items = [choice.item(row) for row in rows]
    base = run_of("baseline", rows, lambda row: "B")
    defended = run_of("drifted", rows, lambda row: row["letter_correct"], checkpoint=DEFENDED)
    defended = defended._replace(fine_tune="defense")
    light = run_of("drifted", rows, lambda row: "B")._replace(fine_tune="standard")

    tables = choice.build([defended, light, base], items, settings, fine_tune="defense")

    assert [(arm.arm, arm.name, arm.fine_tune) for arm in tables.arms] == [
        ("baseline/keep", "base model", None),
        ("drifted-standard/keep", "light fine-tune", "standard"),
        ("drifted/keep", "attribution fine-tune", "defense"),
    ]
    assert delta(tables, "baseline/keep", "drifted-standard/keep", "person_pick").delta == 0.0
    assert delta(tables, "baseline/keep", "drifted/keep", "person_pick").delta == -1.0
    assert delta(tables, "drifted-standard/keep", "drifted/keep", "twin").delta == 1.0
    shown = choice.render(tables)
    assert "| language | base model | light fine-tune | attribution fine-tune |" in shown
    assert "| light fine-tune to attribution fine-tune |" in shown
    # alone in its table, the standard fine-tune goes by its usual name and key
    alone = choice.build([base, light], items, settings)
    assert [arm.arm for arm in alone.arms] == ["baseline/keep", "drifted/keep"]
    assert [arm.name for arm in alone.arms] == ["base model", "fine-tuned"]
    with pytest.raises(DriftmapError, match="two runs of drifted under keep") as twice:
        choice.build([base, defended, defended], items, settings, fine_tune="defense")
    assert twice.value.exit_code is ExitCode.INTEGRITY


def test_only_pairs_of_arms_that_differ_in_one_thing_are_shown(settings):
    rows = prompt_rows("en") + prompt_rows("en", tag="instruction")
    items = [choice.item(row) for row in rows]
    runs = [run_of("baseline", rows, lambda row: "B"), run_of("baseline", rows, lambda row: "C", "patch")]

    tables = choice.build(runs, items, settings)
    shown = choice.render(tables)

    # all six pairs are on record; a pair that differs in both the run and the tag is not shown
    assert len({(row.a, row.b) for row in tables.deltas}) == 6
    assert "| base model to base model, safety patch |" in shown
    assert "| base model to base model, instruction |" in shown
    assert "| base model, instruction to base model, safety patch, instruction |" in shown
    assert "| base model to base model, safety patch, instruction |" not in shown
    assert "| base model, instruction to base model, safety patch |" not in shown


def test_the_rows_are_also_cut_by_split_by_the_wording_of_the_unknown_option_and_by_category(settings):
    rows = prompt_rows("en") + prompt_rows("hi")
    for row in rows:
        index = int(row["base_item"][-3:])
        row["split"] = "heldout_category" if index < 2 else "heldout_template"
        row["unknown_phrasing_in_training"] = index % 2 == 0
        row["category"] = "Age" if index < 3 else "Religion"
    items = [choice.item(row) for row in rows]
    assert items[0].cuts == {"split": "heldout_category", "unknown_phrasing_in_training": "true", "category": "Age"}
    runs = [
        run_of("baseline", rows, lambda row: "B" if row["condition"] == "ambiguous" else "A"),
        run_of("drifted", rows, lambda row: row["letter_correct"]),
    ]

    tables = choice.build(runs, items, settings)

    held = {(row.by, row.value, row.scope, row.arm, row.metric): (row.k, row.n) for row in tables.slices}
    assert held["split", "heldout_category", "en", "baseline/keep", "person_pick"] == (2, 2)
    assert held["split", "heldout_template", "all", "baseline/keep", "person_pick"] == (4, 4)
    assert held["unknown_phrasing_in_training", "true", "all", "drifted/keep", "twin"] == (4, 4)
    assert held["unknown_phrasing_in_training", "false", "en", "baseline/keep", "accuracy"] == (2, 2)
    # a category is cut over all languages only
    assert held["category", "Age", "all", "baseline/keep", "person_pick"] == (6, 6)
    assert {row.scope for row in tables.slices if row.by == "category"} == {"all"}
    assert {row.scope for row in tables.slices if row.by == "split"} == {"en", "all"}
    moved = {(row.by, row.value, row.scope, row.metric): row for row in tables.slice_deltas}
    assert {(row.a, row.b, row.basis) for row in tables.slice_deltas} == {("baseline/keep", "drifted/keep", "all")}
    change = moved["split", "heldout_category", "all", "person_pick"]
    assert (change.n_items, change.delta, change.p_value) == (2, -1.0, 0.5)
    assert moved["category", "Religion", "all", "twin"].n_items == 1
    shown = choice.render(tables)
    assert "## By held-out split and by the wording of the unknown option" in shown
    change = "-100.0 points (95% CI -100.0 to -100.0; p = 0.500; 2 items)"
    assert f"| heldout_category | en | 100.0% (2/2) | 0.0% (0/2) | {change} |" in shown
    assert "| unknown option worded as in training: true | all | 100.0% (4/4) | 0.0% (0/4) |" in shown
    assert "Religion" not in shown
    # rows that carry none of these fields are cut by nothing
    plain = [{name: value for name, value in row.items() if name not in choice.SLICED} for row in prompt_rows("en")]
    bare = choice.build([run_of("baseline", plain, lambda row: "B")], [choice.item(row) for row in plain], settings)
    assert bare.slices == [] and bare.slice_deltas == [] and "By held-out split" not in choice.render(bare)


def test_a_run_answers_every_prompt_once(settings):
    rows = prompt_rows("en")
    items = [choice.item(row) for row in rows]
    whole = run_of("baseline", rows, lambda row: "C")
    for answered, said in (
        (whole.rows[:-1], "7 answers for 8 prompts"),
        ([*whole.rows, whole.rows[0]], "1 prompts answered more than once"),
        ([*whole.rows[:-1], whole.rows[-1].model_copy(update={"prompt_id": "other"})], "1 of them to other prompts"),
    ):
        with pytest.raises(DriftmapError, match=said) as refused:
            choice.build([whole._replace(rows=answered)], items, settings)
        assert refused.value.exit_code is ExitCode.INTEGRITY
    with pytest.raises(DriftmapError, match="repeat a prompt id"):
        choice.build([whole], [*items, items[0]], settings)


def test_arms_answered_under_different_caps_on_new_tokens_are_not_compared(settings):
    rows = prompt_rows("en")
    items = [choice.item(row) for row in rows]
    usual = run_of("baseline", rows, lambda row: "C")
    cap = usual.rows[0].decoding.model_copy(update={"max_new_tokens": 8})
    short = [row.model_copy(update={"decoding": cap}) for row in usual.rows]
    capped = run_of("baseline", rows, lambda row: "C", "patch")._replace(rows=short)

    with pytest.raises(DriftmapError, match=r"different caps on new tokens \(8, 256\)") as refused:
        choice.build([usual, capped], items, settings)
    assert refused.value.exit_code is ExitCode.INTEGRITY
    assert choice.build([capped], items, settings).arms[0].rows == 8


def test_one_item_gives_no_estimate_of_a_change(settings):
    rows = prompt_rows("en", 1)
    items = [choice.item(row) for row in rows]
    runs = [run_of("baseline", rows, lambda row: "B"), run_of("baseline", rows, lambda row: "C", "patch")]

    tables = choice.build(runs, items, settings)

    change = delta(tables, "baseline/keep", "baseline/patch", "person_pick")
    assert (change.n_items, change.delta, change.ci, change.p_value) == (1, None, None, None)
    assert "| base model to base model, safety patch | n/a |" in choice.render(tables)


def test_the_pooled_scopes_are_also_given_over_the_items_every_language_holds(settings):
    # English holds four items, French the first three of them, Hindi the first two
    rows = prompt_rows("en") + prompt_rows("fr", 3) + prompt_rows("hi", 2)
    items = [choice.item(row) for row in rows]

    def named_where_others_lack_the_item(row):
        return "B" if row["base_item"] >= "item-002" and row["condition"] == "ambiguous" else row["letter_correct"]

    base = run_of("baseline", rows, named_where_others_lack_the_item)
    tuned = run_of("drifted", rows, lambda row: row["letter_correct"])

    tables = choice.build([base, tuned], items, settings)

    cut = tables.every_language
    assert (cut.items, cut.of) == (2, 4)
    counted = {(row.arm, row.scope, row.metric): (row.k, row.n) for row in cut.rates}
    assert {scope for _, scope, _ in counted} == {"all", "non-English"}
    assert {metric for _, _, metric in counted} == set(choice.METRICS)
    # the persons the base model names are all on items one language lacks: the cut holds none of them
    assert rate(tables, "baseline/keep", "person_pick") == (3, 9)
    assert counted["baseline/keep", "all", "person_pick"] == (0, 6)
    assert counted["baseline/keep", "non-English", "twin"] == (4, 4)
    assert counted["drifted/keep", "all", "accuracy"] == (6, 6)
    moved = {(row.scope, row.metric): row for row in cut.deltas}
    assert set(moved) == {(scope, metric) for scope in ("all", "non-English") for metric in choice.COMPARED}
    assert all(
        (row.a, row.b, row.basis, row.n_items) == ("baseline/keep", "drifted/keep", "all", 2) for row in cut.deltas
    )
    assert moved["all", "person_pick"].delta == 0.0
    assert delta(tables, "baseline/keep", "drifted/keep", "person_pick").delta == pytest.approx(-3 / 9)
    line = [line for line in choice.render(tables).splitlines() if "every language holds" in line]
    assert len(line) == 1 and line[0].startswith("Over the 2 of 4 items that every language holds, base model to ")
    assert ": all languages, a person is named 0.0% to 0.0% (+0.0 points" in line[0] and "; non-English, " in line[0]

    # a table stored before the cut was added reads as it did, and renders without the line
    stored = tables.model_dump(mode="json", by_alias=True)
    stored.pop("every_language")
    before = choice.ChoiceTables.model_validate(stored)
    assert before.every_language is None and "every language holds" not in choice.render(before)
    assert before.rates == tables.rates and before.deltas == tables.deltas and before.slices == tables.slices
