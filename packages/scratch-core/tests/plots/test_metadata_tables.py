import pytest
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from conversion.data_formats import MarkMetadata
from plots.metadata_tables import (
    metadata_to_table_data,
    get_col_widths,
    get_bounding_box,
    get_metadata_dimensions,
    draw_metadata_box,
)


class TestMetadataToTableData:
    def test_simple_metadata(self):
        metadata = {"Key": "Value"}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result == [["Key:", "Value"]]

    def test_wrapping_long_values(self):
        metadata = {"Key": "A" * 100}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert len(result) > 1
        assert result[0][0] == "Key:"
        assert result[1][0] == ""  # Continuation has empty key

    def test_empty_value(self):
        metadata = {"Key": ""}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result == [["Key:", ""]]

    def test_preserves_order(self):
        metadata = {"First": "1", "Second": "2", "Third": "3"}
        result = metadata_to_table_data(metadata, wrap_width=40)
        keys = [row[0] for row in result]
        assert keys == ["First:", "Second:", "Third:"]

    def test_non_string_values_converted(self):
        metadata = {"Number": 42, "Float": 3.14}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result[0] == ["Number:", "42"]
        assert result[1] == ["Float:", "3.14"]

    def test_empty_key_skips_colon(self):
        metadata = {"Key": "Value", "": ""}
        result = metadata_to_table_data(metadata, wrap_width=40)
        assert result[1][0] == ""  # No colon for empty key

    def test_wrapping_produces_continuation_rows(self):
        metadata = {"Short": "v", "Long": "A" * 80}
        result = metadata_to_table_data(metadata, wrap_width=25)
        assert result[0][0] == "Short:"
        assert result[1][0] == "Long:"
        # Continuation rows have empty key
        for row in result[2:]:
            assert row[0] == ""

    def test_multiple_keys_with_wrapping(self):
        metadata = {"A": "x" * 60, "B": "y" * 60}
        result = metadata_to_table_data(metadata, wrap_width=25)
        # Total rows should be more than 2 due to wrapping
        assert len(result) > 2


class TestGetColWidths:
    def test_returns_two_widths(self):
        table_data = [["Key:", "Value"]]
        key_w, val_w = get_col_widths(0.06, table_data)
        assert key_w > 0
        assert val_w > 0

    def test_widths_fit_within_margins(self):
        table_data = [["Key:", "Value"]]
        side_margin = 0.10
        key_w, val_w = get_col_widths(side_margin, table_data)
        assert key_w + val_w == pytest.approx(1.0 - 2 * side_margin, abs=0.01)

    def test_long_keys_get_more_width(self):
        short = [["K:", "Value"]]
        long = [["Very Long Key Name:", "V"]]
        key_w_short, _ = get_col_widths(0.06, short)
        key_w_long, _ = get_col_widths(0.06, long)
        assert key_w_long > key_w_short

    def test_key_ratio_clamped_between_35_and_50_percent(self):
        # Very short key compared to long value
        table_data = [["K:", "A" * 100]]
        key_w, val_w = get_col_widths(0.0, table_data)
        total = key_w + val_w
        assert key_w / total >= 0.35 - 0.001


class TestGetBoundingBox:
    def test_returns_bbox(self):
        table_data = [["Key:", "Value"]] * 3
        result = get_bounding_box(0.06, table_data)
        assert isinstance(result, Bbox)

    def test_fewer_rows_give_more_height_per_row(self):
        few = get_bounding_box(0.06, [["K:", "V"]] * 3)
        many = get_bounding_box(0.06, [["K:", "V"]] * 12)
        # Fewer rows -> more generous spacing per row (height/n_rows)
        assert few.height / 3 > many.height / 12

    def test_bbox_centered_vertically(self):
        table_data = [["K:", "V"]] * 5
        bbox = get_bounding_box(0.06, table_data)
        center_y = bbox.y0 + bbox.height / 2
        assert center_y == pytest.approx(0.5, abs=0.05)

    def test_respects_side_margin(self):
        margin = 0.10
        bbox = get_bounding_box(margin, [["K:", "V"]])
        assert bbox.x0 == pytest.approx(margin)
        assert bbox.width == pytest.approx(1.0 - 2 * margin)


class TestGetMetadataDimensions:
    def test_returns_rows_and_ratio(self):
        meta = MarkMetadata(
            case_id="V",
            firearm_id="V",
            specimen_id="V",
            measurement_id="V",
            mark_id="V",
        )
        rows, ratio = get_metadata_dimensions(meta, meta, wrap_width=25)
        assert rows >= 1
        assert ratio >= 0.12

    def test_uses_max_of_both_metadata(self):
        short = MarkMetadata(
            case_id="A",
            firearm_id="B",
            specimen_id="C",
            measurement_id="D",
            mark_id="E",
        )
        long = MarkMetadata(
            case_id="A" * 80,
            firearm_id="B" * 80,
            specimen_id="C" * 80,
            measurement_id="D" * 80,
            mark_id="E" * 80,
        )

        rows_short, _ = get_metadata_dimensions(short, short, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long, long, wrap_width=25)
        assert rows_long > rows_short, "fixture setup: long must wrap more than short"

        # The larger of the two dictionaries wins, whichever argument it is.
        rows_mixed, _ = get_metadata_dimensions(short, long, wrap_width=25)
        rows_mixed_reversed, _ = get_metadata_dimensions(long, short, wrap_width=25)
        assert rows_mixed == rows_long
        assert rows_mixed_reversed == rows_long

    def test_ratio_has_minimum_floor(self):
        tiny = MarkMetadata(
            case_id="A",
            firearm_id="B",
            specimen_id="C",
            measurement_id="D",
            mark_id="E",
        )
        rows, ratio = get_metadata_dimensions(tiny, tiny, wrap_width=25)
        assert rows * 0.022 < 0.12, "fixture setup: should be below the floor"
        assert ratio == pytest.approx(0.12)

    def test_wrapping_increases_rows(self):
        short = MarkMetadata(
            case_id="Short",
            firearm_id="Short",
            specimen_id="Short",
            measurement_id="Short",
            mark_id="Short",
        )
        long = MarkMetadata(
            case_id="A" * 100,
            firearm_id="A" * 100,
            specimen_id="A" * 100,
            measurement_id="A" * 100,
            mark_id="A" * 100,
        )
        rows_short, _ = get_metadata_dimensions(short, short, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long, long, wrap_width=25)
        assert rows_long > rows_short


class TestDrawMetadataBox:
    def test_draws_table_with_border(self):
        fig, ax = plt.subplots()
        metadata = {"Key": "Value", "Other": "Data"}
        draw_metadata_box(ax, metadata, title="Test")
        assert ax.get_title() == "Test"
        plt.close(fig)

    def test_without_border(self):
        fig, ax = plt.subplots()
        draw_metadata_box(ax, {"K": "V"}, draw_border=False)
        for spine in ax.spines.values():
            assert not spine.get_visible()
        plt.close(fig)

    def test_no_title(self):
        fig, ax = plt.subplots()
        draw_metadata_box(ax, {"K": "V"}, title=None)
        assert ax.get_title() == ""
        plt.close(fig)

    def test_empty_key_in_metadata(self):
        fig, ax = plt.subplots()
        metadata = {"Key": "Value", "": "", "Other": "Data"}
        draw_metadata_box(ax, metadata)
        plt.close(fig)
