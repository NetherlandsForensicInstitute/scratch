from enum import StrEnum, auto

import pytest
from hypothesis import given
from hypothesis import strategies as st

from models import validate_file_extension


class SampleExtensions(StrEnum):
    """Sample enum for file extensions used in tests."""

    X3P = auto()
    PNG = auto()
    TXT = auto()


@given(
    path=st.one_of(
        st.just(""),  # No path
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).map(lambda x: f"/{x}/"),  # Absolute path
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).map(lambda x: f"{x}/"),  # Relative path
    ),
    basename=st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd", "P"), min_codepoint=45, max_codepoint=122
        ).filter(lambda c: c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"),
        min_size=1,
        max_size=20,
    ),
    extension=st.sampled_from(SampleExtensions),
)
def test_validate_file_extension_valid(path: str, basename: str, extension: str) -> None:
    """Test that valid file extensions pass validation with various paths using property-based testing."""
    # Arrange
    filename = f"{path}{basename}{extension}"

    # Act
    result = validate_file_extension(filename, SampleExtensions)

    # Assert
    assert result == filename


@given(
    invalid_filename=st.one_of(
        # Empty string
        st.just(""),
        # Files with no extension
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ),
        # Files with invalid extensions
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from([".pdf", ".jpg", ".docx", ".zip", ".tar"]).map(
                lambda ext: f"{basename}{ext}"
            )
        ),
        # Files with wrong case extensions
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from([".X3P", ".PNG", ".TXT", ".X3p", ".Png"]).map(
                lambda ext: f"{basename}{ext}"
            )
        ),
        # Partial extension matches (extension + extra characters)
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).flatmap(
            lambda basename: st.sampled_from(SampleExtensions).flatmap(
                lambda ext: st.text(
                    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
                    min_size=1,
                    max_size=3,
                ).map(lambda suffix: f"{basename}{ext}{suffix}")
            )
        ),
    )
)
def test_validate_file_extension_invalid(invalid_filename: str) -> None:
    """Test that invalid filenames raise ValueError using property-based testing."""
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        validate_file_extension(invalid_filename, SampleExtensions)

    # Assert
    error_message = str(exc_info.value)
    assert "unsupported file type" in error_message
    assert "try: x3p, png, txt" in error_message
