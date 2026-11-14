import pytest

from driftmap.figures import draw
from driftmap.table.breakdown import build as build_breakdown
from tests.fakes.breakdown import HATE, lay, prompts, stored_variants, two_conditions, variant_row
from tests.fakes.labels import make_settings
from tests.fakes.published import two_column_tables

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


STEMS = (
    "f1_harmful_by_language",
    "f2_category_by_language",
    "f3_three_axes",
    "f4_harm_categories",
    "f5_transitions",
    "f6_truncation",
    "f7_script_variants",
    "f8_safety_patch",
)


def test_the_figures_are_written_as_png_and_svg_and_are_the_same_on_every_run(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    breakdown = two_conditions(tmp_path, settings)

    written = draw.draw_all(tables, breakdown, settings, tmp_path / "first" / "figures")

    # the variants and the patch have no data here
    assert [path.name for path in written] == [f"{stem}.{kind}" for stem in STEMS[:6] for kind in ("png", "svg")]
    assert all(path.parent == tmp_path / "first" / "figures" for path in written)
    for path in written:
        content = path.read_bytes()
        assert len(content) > 2000, path.name
        if path.suffix == ".png":
            # 9 inches at 160 dpi
            assert content.startswith(PNG) and int.from_bytes(content[16:20]) == 1440, path.name
        else:
            assert content.lstrip().startswith(b"<?xml") and b"<dc:date>" not in content, path.name
    again = draw.draw_all(tables, breakdown, settings, tmp_path / "second")
    assert [path.read_bytes() for path in again] == [path.read_bytes() for path in written]


def test_a_figure_is_drawn_once_its_data_is_there(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    labels = {"keep": tmp_path / "keep", "patch": tmp_path / "patch"}
    plan = prompts("en", [HATE], 20, {"harmful": 5})
    alone = build_breakdown(labels, {"harmful": lay(labels["keep"], plan, columns=("baseline",))}, [], settings)

    # one column, no stored answers: neither outcomes nor lengths
    written = draw.draw_all(tables, alone, settings, tmp_path / "out")
    assert [path.stem for path in written[::2]] == list(STEMS[:4])

    lay(labels["patch"], prompts("en", [HATE], 20, {"harmful": 2}))
    patched = build_breakdown(labels, {"harmful": lay(labels["keep"], plan)}, [], settings)
    rows = [variant_row(column, "all-Latn", 0.10, 0.25, ci=(0.09, 0.21)) for column in ("baseline", "drifted")]
    written = draw.draw_all(tables, patched, settings, tmp_path / "more", draw.data.variants(stored_variants(*rows)))
    assert [path.stem for path in written[::2]] == [*STEMS[:5], *STEMS[6:]]
    assert all(path.stat().st_size > 2000 for path in written)
