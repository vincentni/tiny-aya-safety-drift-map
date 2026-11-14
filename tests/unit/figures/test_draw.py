import pytest

from driftmap.figures import draw
from driftmap.table.breakdown import build as build_breakdown
from tests.fakes.breakdown import HATE, lay, prompts, two_conditions
from tests.fakes.labels import make_settings
from tests.fakes.published import two_column_tables

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def test_the_five_figures_are_written_as_png_and_svg_and_are_the_same_on_every_run(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    breakdown = two_conditions(tmp_path, settings)

    written = draw.draw_all(tables, breakdown, settings, tmp_path / "first" / "figures")

    assert [path.name for path in written] == [f"{stem}.{kind}" for stem in draw.STEMS for kind in ("png", "svg")]
    assert len(draw.STEMS) == 5 and all(path.parent == tmp_path / "first" / "figures" for path in written)
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


def test_a_breakdown_of_one_column_gets_every_figure_but_the_outcomes(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    labels = {"keep": tmp_path / "keep"}
    plan = prompts("en", [HATE], 20, {"harmful": 5})
    alone = build_breakdown(labels, {"harmful": lay(labels["keep"], plan, columns=("baseline",))}, [], settings)

    written = draw.draw_all(tables, alone, settings, tmp_path / "out")

    assert [path.stem for path in written[::2]] == list(draw.STEMS[:4])
