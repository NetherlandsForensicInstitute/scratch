"""Unit tests for the mark serialization module."""

import json
from pathlib import Path

import numpy as np

from container_models.scan_image import ScanImage
from conversion.export.mark import (
    ExportedMarkData,
    ScanImageMetaData,
    _check_file_exists,
    _to_json,
    from_path,
    save,
)
from pydantic import ValidationError
from conversion.data_formats import CropType, Mark, MarkType

import pytest


@pytest.fixture
def sample_mark_with_center(scan_image_rectangular_with_nans: ScanImage) -> Mark:
    """
    Create a sample Mark with explicit center for testing.

    :param scan_image_rectangular_with_nans: ScanImage fixture
    :returns: Mark instance with test data and explicit center
    """
    mark = Mark(
        scan_image=scan_image_rectangular_with_nans,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
        meta_data={"key": "value", "source": "test"},
    )
    mark._center = (10.0, 20.0)
    return mark


@pytest.fixture
def sample_mark_without_center(scan_image_rectangular_with_nans: ScanImage) -> Mark:
    """
    Create a sample Mark without explicit center for testing.

    :param scan_image_rectangular_with_nans: ScanImage fixture
    :returns: Mark instance with computed center
    """
    return Mark(
        scan_image=scan_image_rectangular_with_nans,
        mark_type=MarkType.FIRING_PIN_IMPRESSION,
        crop_type=CropType.CIRCLE,
        meta_data={},
    )


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for file operations.

    :param tmp_path: Pytest tmp_path fixture
    :returns: Path to temporary directory
    """
    return tmp_path


class TestExportedMarkData:
    """Tests for ExportedMarkData validation."""

    def test_valid_exported_mark_data(self) -> None:
        """Test creation with valid data."""
        data = ExportedMarkData(
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
            crop_type=CropType.RECTANGLE,
            center=(1.0, 2.0),
            scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
        )
        assert data.mark_type == MarkType.BREECH_FACE_IMPRESSION
        assert data.crop_type == CropType.RECTANGLE
        assert data.center == (1.0, 2.0)

    def test_mark_type_validation_from_string(self) -> None:
        """Test mark_type validation accepts valid string (case-insensitive)."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type="FIRING_PIN_IMPRESSION",
                crop_type=CropType.CIRCLE,
                center=(1.0, 2.0),
                scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
            )
        )
        assert data.mark_type == MarkType.FIRING_PIN_IMPRESSION

        # Test lowercase also works
        data_lower = ExportedMarkData.model_validate(
            dict(
                mark_type="firing_pin_impression",
                crop_type=CropType.CIRCLE,
                center=(1.0, 2.0),
                scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
            )
        )
        assert data_lower.mark_type == MarkType.FIRING_PIN_IMPRESSION

    def test_mark_type_validation_invalid(self) -> None:
        """Test mark_type validation rejects invalid string."""
        with pytest.raises(ValidationError, match="Unsupported mark type"):
            ExportedMarkData.model_validate(
                dict(
                    mark_type="INVALID_MARK_TYPE",
                    crop_type=CropType.RECTANGLE,
                    center=(1.0, 2.0),
                    scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
                )
            )

    def test_crop_type_validation_from_string(self) -> None:
        """Test crop_type validation accepts valid string (case-insensitive)."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type=MarkType.CHAMBER_IMPRESSION,
                crop_type="ELLIPSE",
                center=(1.0, 2.0),
                scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
            )
        )
        assert data.crop_type == CropType.ELLIPSE

        # Test lowercase also works
        data_lower = ExportedMarkData.model_validate(
            dict(
                mark_type=MarkType.CHAMBER_IMPRESSION,
                crop_type="ellipse",
                center=(1.0, 2.0),
                scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
            )
        )
        assert data_lower.crop_type == CropType.ELLIPSE

    def test_crop_type_validation_invalid(self) -> None:
        """Test crop_type validation rejects invalid string."""
        with pytest.raises(ValidationError, match="Unsupported crop type"):
            ExportedMarkData.model_validate(
                dict(
                    mark_type=MarkType.EXTRACTOR_IMPRESSION,
                    crop_type="INVALID_CROP",
                    center=(1.0, 2.0),
                    scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
                )
            )

    def test_meta_data_default(self) -> None:
        """Test that meta_data defaults to empty dict."""
        data = ExportedMarkData.model_validate(
            dict(
                mark_type=MarkType.EJECTOR_IMPRESSION,
                crop_type=CropType.POLYGON,
                center=(1.0, 2.0),
                scan_image=ScanImageMetaData(scale_x=1.0, scale_y=2.0),
            )
        )
        assert data.meta_data == {}


class TestSaveAndLoad:
    """Tests for save and load operations."""

    def test_save_creates_files(
        self, sample_mark_with_center: Mark, temp_dir: Path
    ) -> None:
        """
        Test that save creates both JSON and NPZ files.

        :param sample_mark_with_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        save(sample_mark_with_center, temp_dir, "test_mark")

        json_file = temp_dir / "test_mark.json"
        npz_file = temp_dir / "test_mark.npz"

        assert json_file.exists()
        assert npz_file.exists()

    def test_save_creates_directory(
        self, sample_mark_with_center: Mark, temp_dir: Path
    ) -> None:
        """
        Test that save creates parent directories if needed.

        :param sample_mark_with_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        nested_dir = temp_dir / "nested" / "dir"
        save(sample_mark_with_center, nested_dir, "test_mark")

        assert nested_dir.exists()
        assert (nested_dir / "test_mark.json").exists()

    def test_roundtrip_save_and_load_with_explicit_center(
        self, sample_mark_with_center: Mark, temp_dir: Path
    ) -> None:
        """
        Test that save and load preserve Mark data with explicit center.

        :param sample_mark_with_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        save(sample_mark_with_center, temp_dir, "test_mark")
        loaded_mark = from_path(temp_dir, "test_mark")

        assert loaded_mark.mark_type == sample_mark_with_center.mark_type
        assert loaded_mark.crop_type == sample_mark_with_center.crop_type
        assert loaded_mark.center == sample_mark_with_center.center
        assert loaded_mark.meta_data == sample_mark_with_center.meta_data
        np.testing.assert_array_equal(
            loaded_mark.scan_image.data, sample_mark_with_center.scan_image.data
        )
        assert (
            loaded_mark.scan_image.scale_x == sample_mark_with_center.scan_image.scale_x
        )
        assert (
            loaded_mark.scan_image.scale_y == sample_mark_with_center.scan_image.scale_y
        )

    def test_roundtrip_save_and_load_with_computed_center(
        self, sample_mark_without_center: Mark, temp_dir: Path
    ) -> None:
        """
        Test that save and load preserve Mark with computed center.

        :param sample_mark_without_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        # Get computed center before save
        original_center = sample_mark_without_center.center

        save(sample_mark_without_center, temp_dir, "test_mark")
        loaded_mark = from_path(temp_dir, "test_mark")

        # The computed center should be saved and then loaded as explicit center
        assert loaded_mark.center == original_center
        assert loaded_mark.mark_type == sample_mark_without_center.mark_type
        assert loaded_mark.crop_type == sample_mark_without_center.crop_type

    def test_from_path_missing_json(self, temp_dir: Path) -> None:
        """Test from_path raises FileNotFoundError when JSON missing.

        :param temp_dir: Temporary directory fixture
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            from_path(temp_dir, "nonexistent")

    def test_from_path_missing_npz(
        self, sample_mark_with_center: Mark, temp_dir: Path
    ) -> None:
        """
        Test from_path raises FileNotFoundError when NPZ missing.

        :param sample_mark_with_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        # Create only JSON file
        json_file = temp_dir / "test_mark.json"
        json_file.write_text(_to_json(sample_mark_with_center))

        with pytest.raises(FileNotFoundError, match="does not exist"):
            from_path(temp_dir, "test_mark")


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_to_json_format_impression_mark(
        self, sample_mark_with_center: Mark
    ) -> None:
        """
        Test JSON serialization format for impression mark.

        :param sample_mark_with_center: Mark fixture
        """
        json_str = _to_json(sample_mark_with_center)
        data = json.loads(json_str)

        assert data["mark_type"] == sample_mark_with_center.mark_type.name
        assert data["crop_type"] == sample_mark_with_center.crop_type.name
        assert data["center"] == list(sample_mark_with_center.center)
        assert (
            data["scan_image"]["scale_x"] == sample_mark_with_center.scan_image.scale_x
        )
        assert data["meta_data"] == sample_mark_with_center.meta_data

    def test_to_json_format_striation_mark(
        self, scan_image_rectangular_with_nans: ScanImage
    ) -> None:
        """
        Test JSON serialization format for striation mark.

        :param scan_image_rectangular_with_nans: ScanImage fixture
        """
        striation_mark = Mark(
            scan_image=scan_image_rectangular_with_nans,
            mark_type=MarkType.EJECTOR_STRIATION,
            crop_type=CropType.POLYGON,
        )
        striation_mark._center = (5.0, 10.0)

        json_str = _to_json(striation_mark)
        data = json.loads(json_str)

        assert data["mark_type"] == "EJECTOR_STRIATION"
        assert data["crop_type"] == "POLYGON"
        assert data["center"] == [5.0, 10.0]

    def test_load_binary_correct_data(
        self, sample_mark_with_center: Mark, temp_dir: Path, filename: str = "test_mark"
    ) -> None:
        """
        Test binary loading returns correct array.

        :param sample_mark_with_center: Mark fixture
        :param temp_dir: Temporary directory fixture
        """
        save(sample_mark_with_center, temp_dir, filename)
        loaded_mark = from_path(temp_dir, filename)
        np.testing.assert_array_equal(
            sample_mark_with_center.scan_image.data, loaded_mark.scan_image.data
        )

    def test_check_file_exists_valid(self, temp_dir: Path) -> None:
        """
        Test _check_file_exists with existing file.

        :param temp_dir: Temporary directory fixture
        """
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        # Should not raise
        _check_file_exists(test_file)

    def test_check_file_exists_missing(self, temp_dir: Path) -> None:
        """
        Test _check_file_exists with missing file.

        :param temp_dir: Temporary directory fixture
        """
        with pytest.raises(FileNotFoundError):
            _check_file_exists(temp_dir / "nonexistent.txt")

    def test_save_and_load_all_mark_types(
        self, scan_image_rectangular_with_nans: ScanImage, temp_dir: Path
    ) -> None:
        """
        Test save and load works for all MarkType enum values.

        :param scan_image_rectangular_with_nans: ScanImage fixture
        :param temp_dir: Temporary directory fixture
        """
        for mark_type in MarkType:
            mark = Mark(
                scan_image=scan_image_rectangular_with_nans,
                mark_type=mark_type,
                crop_type=CropType.RECTANGLE,
            )
            mark._center = (1.0, 20.0)

            filename = f"test_{mark_type.name.lower()}"
            save(mark, temp_dir, filename)
            loaded_mark = from_path(temp_dir, filename)

            assert loaded_mark.mark_type == mark_type

    def test_save_and_load_all_crop_types(
        self, scan_image_rectangular_with_nans: ScanImage, temp_dir: Path
    ) -> None:
        """
        Test save and load works for all CropType enum values.

        :param scan_image_rectangular_with_nans: ScanImage fixture
        :param temp_dir: Temporary directory fixture
        """
        for crop_type in CropType:
            mark = Mark(
                scan_image=scan_image_rectangular_with_nans,
                mark_type=MarkType.FIRING_PIN_IMPRESSION,
                crop_type=crop_type,
            )
            mark._center = (1.0, 2.0)

            filename = f"test_{crop_type.name.lower()}"
            save(mark, temp_dir, filename)
            loaded_mark = from_path(temp_dir, filename)

            assert loaded_mark.crop_type == crop_type
