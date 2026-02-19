import json
from pathlib import PosixPath

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.export.mark import ExportedMarkData, load_mark_from_path, save_mark


@pytest.fixture()
def scan_image(scan_image_with_nans: ScanImage) -> ScanImage:
    """Convenience fixture for a `ScanImage` instance."""
    return scan_image_with_nans


class TestExportedMarkData:
    """Tests for ExportedMarkData validation."""

    def test_valid_mark_data(self):
        """Test creation with valid data."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type="BREECH_FACE_IMPRESSION",
                center=(100.0, 200.0),
                scale_x=1.5e-6,
                scale_y=1.5e-6,
                meta_data={"key": "value"},
            )
        )

        assert data.mark_type == MarkType.BREECH_FACE_IMPRESSION
        assert data.center == (100.0, 200.0)
        assert data.scale_x == 1.5e-6
        assert data.scale_y == 1.5e-6
        assert data.meta_data == {"key": "value"}

    def test_lowercase_mark_type(self):
        """Test that lowercase `mark_type` is correctly converted."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type="chamber_impression",
                center=(0.0, 0.0),
                scale_x=1.0,
                scale_y=1.0,
            )
        )

        assert data.mark_type == MarkType.CHAMBER_IMPRESSION

    def test_invalid_mark_type(self):
        """Test that invalid `mark_type` raises ValueError."""
        with pytest.raises(ValueError, match="Invalid MarkType"):
            ExportedMarkData.model_validate(
                dict(
                    mark_type="INVALID_TYPE",
                    center=(0.0, 0.0),
                    scale_x=1.0,
                    scale_y=1.0,
                )
            )

    def test_invalid_scale_data(self):
        """Test that invalid scale data raises exception."""
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            _ = ExportedMarkData.model_validate(
                dict(
                    mark_type="BREECH_FACE_IMPRESSION",
                    scale_x=0.0,
                    scale_y=0.0,
                    center=(1.0, 1.0),
                )
            )

    def test_default_meta_data(self):
        """Test that `meta_data` defaults to empty dict."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type="EJECTOR_IMPRESSION",
                center=(50.0, 50.0),
                scale_x=2.0e-6,
                scale_y=2.0e-6,
            )
        )

        assert data.meta_data == {}


@pytest.mark.integration
class TestSaveAndLoadMark:
    """Tests for `save_mark` and `load_mark_from_path` functions."""

    def test_save_mark_creates_files(self, tmp_path: PosixPath, scan_image: ScanImage):
        """Test that `save_mark` creates both JSON and NPZ files."""
        impression_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
            meta_data={"test": "data"},
        )

        save_mark(impression_mark, tmp_path / "test_mark")

        assert (tmp_path / "test_mark.json").is_file()
        assert (tmp_path / "test_mark.npz").is_file()

    def test_save_creates_directory(self, tmp_path: PosixPath, scan_image: ScanImage):
        """Test that `save_mark` creates parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "directory"
        impression_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )

        save_mark(impression_mark, nested_path / "test_mark")

        assert nested_path.exists()
        assert (nested_path / "test_mark.json").is_file()
        assert (nested_path / "test_mark.npz").is_file()

    def test_json_content_structure(self, tmp_path: PosixPath, scan_image: ScanImage):
        """Test that saved JSON has correct structure."""
        impression_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.CHAMBER_STRIATION,
            meta_data={"key": "value"},
        )

        save_mark(impression_mark, tmp_path / "test_mark")

        with open(tmp_path / "test_mark.json", "r") as f:
            data = json.load(f)

        assert "mark_type" in data
        assert "center" in data
        assert "scale_x" in data
        assert "scale_y" in data
        assert "meta_data" in data
        assert data["mark_type"] == "CHAMBER_STRIATION"
        assert data["meta_data"] == {"key": "value"}

    def test_load_mark_restores_all_meta_data(
        self, tmp_path: PosixPath, scan_image: ScanImage
    ):
        """Test that `load_mark_from_path` correctly restores all meta-data in a `Mark` object."""
        original_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.EXTRACTOR_IMPRESSION,
            meta_data={"original": "data"},
            center=(123.4, 567.8),
        )

        save_mark(original_mark, tmp_path / "test_mark")
        loaded_mark = load_mark_from_path(tmp_path, "test_mark")

        assert loaded_mark.mark_type == original_mark.mark_type
        assert loaded_mark.center == original_mark.center
        assert loaded_mark.meta_data == original_mark.meta_data
        assert loaded_mark.scan_image.scale_x == original_mark.scan_image.scale_x
        assert loaded_mark.scan_image.scale_y == original_mark.scan_image.scale_y

    def test_load_mark_binary_data_matches(
        self, tmp_path: PosixPath, scan_image: ScanImage
    ):
        """Test that loaded binary data matches original data."""
        original_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.BULLET_GEA_STRIATION,
        )

        save_mark(original_mark, tmp_path / "test_mark")
        loaded_mark = load_mark_from_path(tmp_path, "test_mark")

        np.testing.assert_array_equal(
            loaded_mark.scan_image.data, original_mark.scan_image.data
        )

    def test_load_mark_computed_center(
        self, tmp_path: PosixPath, scan_image: ScanImage
    ):
        """Test loading mark with computed (not explicit) center."""
        original_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.EJECTOR_STRIATION,
        )
        # Don't set explicit center - should compute from image dimensions

        save_mark(original_mark, tmp_path / "test_mark")
        loaded_mark = load_mark_from_path(tmp_path, "test_mark")

        expected_center = (scan_image.height / 2, scan_image.width / 2)
        assert loaded_mark.center == expected_center

    def test_load_mark_missing_json(self, tmp_path: PosixPath):
        """Test that loading raises FileNotFoundError when JSON is missing."""
        with pytest.raises(
            FileNotFoundError, match='File ".*test_mark.json" does not exist'
        ):
            load_mark_from_path(tmp_path, "test_mark")

    def test_load_mark_missing_npz(self, tmp_path: PosixPath, scan_image: ScanImage):
        """Test that loading raises FileNotFoundError when NPZ is missing."""
        mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.FIRING_PIN_DRAG_STRIATION,
        )

        # Save only JSON
        json_path = tmp_path / "test_mark.json"
        json_path.write_text(mark.export())

        with pytest.raises(
            FileNotFoundError, match='File ".*test_mark.npz" does not exist'
        ):
            load_mark_from_path(tmp_path, "test_mark")

    def test_roundtrip_with_complex_metadata(
        self, tmp_path: PosixPath, scan_image: ScanImage
    ):
        """Test save/load roundtrip with complex metadata."""
        complex_meta = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42.5,
            "boolean": True,
            "null": None,
        }

        original_mark = Mark(
            scan_image=scan_image,
            mark_type=MarkType.APERTURE_SHEAR_STRIATION,
            meta_data=complex_meta,
        )

        save_mark(original_mark, tmp_path / "test_mark")
        loaded_mark = load_mark_from_path(tmp_path, "test_mark")

        assert loaded_mark.meta_data == complex_meta

    def test_multiple_marks_in_directory(
        self, tmp_path: PosixPath, scan_image: ScanImage
    ):
        """Test saving and loading multiple marks in the same directory."""
        mark1 = Mark(
            scan_image=scan_image,
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
            meta_data={"id": 1},
        )

        mark2 = Mark(
            scan_image=scan_image,
            mark_type=MarkType.CHAMBER_IMPRESSION,
            meta_data={"id": 2},
        )

        save_mark(mark1, tmp_path / "mark1")
        save_mark(mark2, tmp_path / "mark2")

        loaded_mark1 = load_mark_from_path(tmp_path, "mark1")
        loaded_mark2 = load_mark_from_path(tmp_path, "mark2")

        assert loaded_mark1.mark_type == MarkType.BREECH_FACE_IMPRESSION
        assert loaded_mark2.mark_type == MarkType.CHAMBER_IMPRESSION
        assert loaded_mark1.meta_data["id"] == 1
        assert loaded_mark2.meta_data["id"] == 2
