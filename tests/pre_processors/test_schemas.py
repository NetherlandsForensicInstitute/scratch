from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pre_processors.schemas import EditImage, Filter, Level, SupportedExtension, UploadScan


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def valid_scan_file(tmp_path: Path) -> Path:
    """Create a temporary scan file with valid extension."""
    scan_file = tmp_path / "test_scan.x3p"
    scan_file.touch()
    return scan_file


@pytest.mark.parametrize(
    "extension",
    [ext.value for ext in SupportedExtension],
)
def test_all_supported_extensions(tmp_path: Path, output_dir: Path, extension: str) -> None:
    """Test that all supported extensions are accepted."""
    # Arrange
    scan_file = tmp_path / f"test_scan.{extension}"
    scan_file.touch()

    # Act
    upload_scan = UploadScan(scan_file=scan_file, output_dir=output_dir)

    # Assert
    assert upload_scan.scan_file == scan_file


@given(
    extension=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=10,
    ).filter(lambda ext: ext not in [e.value for e in SupportedExtension])
)
def test_unsupported_extension_raises_error(extension: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test that unsupported file extensions raise ValueError using property-based testing."""
    # Arrange
    tmp_path = tmp_path_factory.mktemp("test_bad_extensions")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scan_file = tmp_path / f"test_scan.{extension}"
    scan_file.touch()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=scan_file, output_dir=output_dir)
    assert "unsupported extension" in str(exc_info.value)


def test_nonexistent_scan_file_raises_error(output_dir: Path) -> None:
    """Test that non-existent scan file raises ValidationError."""
    # Arrange
    nonexistent_file = Path("/nonexistent/path/scan.mat")
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=nonexistent_file, output_dir=output_dir)
    assert "Path does not point to a file" in str(exc_info.value)


def test_nonexistent_output_dir_raises_error(valid_scan_file: Path) -> None:
    """Test that non-existent output directory raises ValidationError."""
    # Arrange
    nonexistent_dir = Path("/nonexistent/output/dir")
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=valid_scan_file, output_dir=nonexistent_dir)
    assert "Path does not point to a directory" in str(exc_info.value)


def test_scan_file_as_directory_raises_error(tmp_path: Path, output_dir: Path) -> None:
    """Test that providing a directory as scan_file raises ValidationError."""
    # Arrange
    directory = tmp_path / "not_a_file"
    directory.mkdir()
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=directory, output_dir=output_dir)
    assert "Path does not point to a file" in str(exc_info.value)


def test_output_dir_as_file_raises_error(tmp_path: Path, valid_scan_file: Path) -> None:
    """Test that providing a file as output_dir raises ValidationError."""
    # Arrange
    file_not_dir = tmp_path / "not_a_directory.txt"
    file_not_dir.touch()
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(scan_file=valid_scan_file, output_dir=file_not_dir)
    assert "Path does not point to a directory" in str(exc_info.value)


def test_model_is_frozen(valid_scan_file: Path, output_dir: Path) -> None:
    """Test that UploadScan instances are immutable (frozen)."""
    # Arrange
    upload_scan = UploadScan(
        scan_file=valid_scan_file,
        output_dir=output_dir,
    )
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        upload_scan.scan_file = valid_scan_file
    assert "Instance is frozen" in str(exc_info.value)


def test_extra_fields_forbidden(valid_scan_file: Path, output_dir: Path) -> None:
    """Test that extra fields are not allowed."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        UploadScan(
            scan_file=valid_scan_file,
            output_dir=output_dir,
            extra_field="not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


# EditImage tests


def test_edit_image_with_defaults(valid_scan_file: Path) -> None:
    """Test EditImage with only required field and default values."""
    # Act
    edit_image = EditImage(parsed_file=valid_scan_file)

    # Assert
    assert edit_image.parsed_file == valid_scan_file
    assert edit_image.sampling == 4  # noqa: PLR2004
    assert edit_image.level is None
    assert edit_image.filter is None
    assert edit_image.zoom is False
    assert edit_image.marks is None


@pytest.mark.parametrize(
    "level",
    [level for level in Level],
)
def test_edit_image_all_level_values(valid_scan_file: Path, level: Level) -> None:
    """Test that all Level enum values are accepted."""
    # Act
    edit_image = EditImage(parsed_file=valid_scan_file, level=level)

    # Assert
    assert edit_image.level == level


@pytest.mark.parametrize(
    "filter_value",
    [filter_val for filter_val in Filter],
)
def test_edit_image_all_filter_values(valid_scan_file: Path, filter_value: Filter) -> None:
    """Test that all Filter enum values are accepted."""
    # Act
    edit_image = EditImage(parsed_file=valid_scan_file, filter=filter_value)

    # Assert
    assert edit_image.filter == filter_value


def test_edit_image_nonexistent_file_raises_error() -> None:
    """Test that non-existent parsed file raises ValidationError."""
    # Arrange
    nonexistent_file = Path("/nonexistent/path/parsed.json")

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(parsed_file=nonexistent_file)
    assert "Path does not point to a file" in str(exc_info.value)


def test_edit_image_directory_raises_error(tmp_path: Path) -> None:
    """Test that providing a directory as parsed_file raises ValidationError."""
    # Arrange
    directory = tmp_path / "not_a_file"
    directory.mkdir()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(parsed_file=directory)
    assert "Path does not point to a file" in str(exc_info.value)


@given(
    level=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=10,
    ).filter(lambda val: val not in [lv.value for lv in Level])
)
def test_edit_image_invalid_level_raises_error(level: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test that invalid level value raises ValidationError using property-based testing."""
    # Arrange
    tmp_path = tmp_path_factory.mktemp("test_invalid_level")
    valid_scan_file = tmp_path / "test_scan.x3p"
    valid_scan_file.touch()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(parsed_file=valid_scan_file, level=level)  # type: ignore[arg-type]
    assert "Input should be" in str(exc_info.value)


@given(
    filter_value=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        min_size=1,
        max_size=10,
    ).filter(lambda val: val not in [fv.value for fv in Filter])
)
def test_edit_image_invalid_filter_raises_error(filter_value: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test that invalid filter value raises ValidationError using property-based testing."""
    # Arrange
    tmp_path = tmp_path_factory.mktemp("test_invalid_filter")
    valid_scan_file = tmp_path / "test_scan.x3p"
    valid_scan_file.touch()

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(parsed_file=valid_scan_file, filter=filter_value)  # type: ignore[arg-type]
    assert "Input should be" in str(exc_info.value)


def test_edit_image_invalid_sampling_type_raises_error(valid_scan_file: Path) -> None:
    """Test that invalid sampling type raises ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(parsed_file=valid_scan_file, sampling="not_an_int")  # type: ignore[arg-type]
    assert "Input should be a valid integer" in str(exc_info.value)


def test_edit_image_with_valid_marks(valid_scan_file: Path) -> None:
    """Test EditImage with valid marks array."""
    # Arrange
    marks = np.array([[10.5, 20.3], [30.1, 40.8], [50.2, 60.9]], dtype=np.float64)

    # Act
    edit_image = EditImage(parsed_file=valid_scan_file, marks=marks)

    # Assert
    assert edit_image.marks is not None
    assert np.array_equal(edit_image.marks, marks)
    assert edit_image.marks.shape[1] == 2  # noqa: PLR2004 # type: ignore


def test_edit_image_model_is_frozen(valid_scan_file: Path) -> None:
    """Test that EditImage instances are immutable (frozen)."""
    # Arrange
    edit_image = EditImage(parsed_file=valid_scan_file)

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        edit_image.sampling = 10
    assert "Instance is frozen" in str(exc_info.value)


def test_edit_image_extra_fields_forbidden(valid_scan_file: Path) -> None:
    """Test that extra fields are not allowed."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        EditImage(
            parsed_file=valid_scan_file,
            extra_field="not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


@given(sampling=st.integers(min_value=1, max_value=100))
def test_edit_image_custom_sampling_values(sampling: int, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test EditImage with different sampling values using property-based testing."""
    # Arrange
    tmp_path = tmp_path_factory.mktemp("test_sampling")
    valid_scan_file = tmp_path / "test_scan.x3p"
    valid_scan_file.touch()

    # Act
    edit_image = EditImage(parsed_file=valid_scan_file, sampling=sampling)

    # Assert
    assert edit_image.sampling == sampling


@pytest.mark.parametrize(
    "zoom",
    [False, True],
)
def test_edit_image_zoom_boolean_values(valid_scan_file: Path, zoom: bool) -> None:
    """Test EditImage with boolean zoom values."""
    # Act
    edit_image = EditImage(parsed_file=valid_scan_file, zoom=zoom)

    # Assert
    assert edit_image.zoom is zoom
