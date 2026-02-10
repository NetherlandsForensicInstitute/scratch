import pytest
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from conversion.plots.utils import (
    draw_metadata_box,
    get_bounding_box,
    get_col_widths,
    get_height_ratios,
    get_metadata_dimensions,
    metadata_to_table_data,
)


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


class TestGetHeightRatios:
    def test_returns_four_ratios(self):
        ratios = get_height_ratios(0.15)
        assert len(ratios) == 4

    def test_ratios_sum_to_one(self):
        ratios = get_height_ratios(0.15)
        assert sum(ratios) == pytest.approx(1.0)

    def test_larger_row0_increases_first_ratio(self):
        small = get_height_ratios(0.10)
        large = get_height_ratios(0.30)
        assert large[0] > small[0]

    def test_fixed_rows_decrease_with_larger_row0(self):
        small = get_height_ratios(0.10)
        large = get_height_ratios(0.30)
        # Rows 1-3 should get smaller fraction as row0 grows
        assert large[1] < small[1]


class TestGetMetadataDimensions:
    def test_returns_rows_and_ratio(self):
        meta = {"Key": "Value"}
        rows, ratio = get_metadata_dimensions(meta, meta, wrap_width=25)
        assert rows >= 1
        assert ratio >= 0.12

    def test_more_keys_increase_rows(self):
        short = {"A": "1"}
        long = {"A": "1", "B": "2", "C": "3", "D": "4"}
        rows_short, _ = get_metadata_dimensions(short, short, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long, long, wrap_width=25)
        assert rows_long > rows_short

    def test_uses_max_of_both_metadata(self):
        short = {"A": "1"}
        long = {"A": "1", "B": "2", "C": "3"}
        rows, _ = get_metadata_dimensions(short, long, wrap_width=25)
        rows_rev, _ = get_metadata_dimensions(long, short, wrap_width=25)
        assert rows == rows_rev

    def test_wrapping_increases_rows(self):
        short_val = {"Key": "Short"}
        long_val = {"Key": "A" * 100}
        rows_short, _ = get_metadata_dimensions(short_val, short_val, wrap_width=25)
        rows_long, _ = get_metadata_dimensions(long_val, long_val, wrap_width=25)
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


class TestMetadataToTableDataExtended:
    """Additional tests for metadata_to_table_data changes."""

    def test_key_row_indices_with_wrapping(self):
        metadata = {"Short": "v", "Long": "A" * 80}
        table_data, key_row_indices = metadata_to_table_data(metadata, wrap_width=25)
        # "Short" is row 0, "Long" is row 1 (with continuation rows after)
        assert 0 in key_row_indices
        assert 1 in key_row_indices
        # Continuation rows should NOT be in key_row_indices
        for i in range(2, len(table_data)):
            assert i not in key_row_indices

    def test_multiple_keys_with_wrapping(self):
        metadata = {"A": "x" * 60, "B": "y" * 60}
        table_data, key_row_indices = metadata_to_table_data(metadata, wrap_width=25)
        assert len(key_row_indices) == 2
        # Total rows should be more than 2 due to wrapping
        assert len(table_data) > 2
