"""Tests for Scratch 2.0 to 3.0 field migration."""

from typing import Any

import pytest
from numpy import array, array_equal, ndarray

from parsers.matfiles import migrate_scratch_2_to_3_fields


@pytest.mark.parametrize(
    "input_data, expected_data",
    [
        # Case 1: Both exist with matching types, matching coordinates (no change)
        pytest.param(
            {
                "selection_type": "rectangle",
                "select_coordinates": array([0, 0, 100, 100]),
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),
            },
            {
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),
            },
            id="both_exist_matching_types_matching_coords",
        ),
        # Case 2: Both exist with matching types, different coordinates (use select)
        pytest.param(
            {
                "selection_type": "rectangle",
                "select_coordinates": array([10, 10, 90, 90]),
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),
            },
            {
                "crop_type": "rectangle",
                "crop_coordinates": array([10, 10, 90, 90]),  # Updated to select
            },
            id="both_exist_matching_types_different_coords",
        ),
        # Case 3: Both exist with matching types, select is empty (keep crop)
        pytest.param(
            {
                "selection_type": "rectangle",
                "select_coordinates": [],
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),
            },
            {
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),  # Unchanged
            },
            id="both_exist_matching_types_select_empty",
        ),
        # Case 4: Both exist with matching types, crop is empty (use select)
        pytest.param(
            {
                "selection_type": "rectangle",
                "select_coordinates": array([10, 10, 90, 90]),
                "crop_type": "rectangle",
                "crop_coordinates": [],
            },
            {
                "crop_type": "rectangle",
                "crop_coordinates": array([10, 10, 90, 90]),  # Updated to select
            },
            id="both_exist_matching_types_crop_empty",
        ),
        # Case 5: Both exist with different types (copy selection to crop)
        pytest.param(
            {
                "selection_type": "circle",
                "select_coordinates": array([50, 50, 25]),
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100, 100]),
            },
            {
                "crop_type": "circle",  # Updated to selection_type
                "crop_coordinates": array(
                    [50, 50, 25]
                ),  # Updated to select_coordinates
            },
            id="both_exist_different_types",
        ),
        # Case 6: Only selection exists (migrate from Scratch 2.0)
        pytest.param(
            {
                "selection_type": "polygon",
                "select_coordinates": array([0, 0, 100, 0, 100, 100, 0, 100]),
            },
            {
                "crop_type": "polygon",  # Copied from selection_type
                "crop_coordinates": array(
                    [0, 0, 100, 0, 100, 100, 0, 100]
                ),  # Copied from select_coordinates
            },
            id="only_selection_exists",
        ),
        # Case 7: Neither exists (do nothing)
        pytest.param(
            {"other_field": "some_value"},
            {"other_field": "some_value"},
            id="neither_exists",
        ),
        # Case 8: Both exist with matching types, different lengths (use select)
        pytest.param(
            {
                "selection_type": "rectangle",
                "select_coordinates": array([10, 10, 90, 90]),
                "crop_type": "rectangle",
                "crop_coordinates": array([0, 0, 100]),  # Different length
            },
            {
                "crop_type": "rectangle",
                "crop_coordinates": array([10, 10, 90, 90]),  # Updated to select
            },
            id="both_exist_matching_types_different_lengths",
        ),
    ],
)
def test_migrate_scratch_2_to_3_fields(
    input_data: dict[str, Any],
    expected_data: dict[str, Any],
) -> None:
    """Test migration of Scratch 2.0 fields to Scratch 3.0 format."""
    data = migrate_scratch_2_to_3_fields(input_data)

    # Check all expected fields
    assert all(
        data[key] == value
        for key, value in expected_data.items()
        if not isinstance(value, ndarray)
    )
    assert all(
        array_equal(data[key], value)
        for key, value in expected_data.items()
        if isinstance(value, ndarray)
    )


def test_migrate_with_numpy_array_strings():
    """Test migration when types are stored as numpy arrays (common in MAT files)."""
    input_data = {
        "selection_type": array(["rectangle"]),  # As numpy array
        "select_coordinates": array([10, 10, 90, 90]),
        "crop_type": array(["rectangle"]),  # As numpy array
        "crop_coordinates": array([0, 0, 100, 100]),
    }

    data = migrate_scratch_2_to_3_fields(input_data)

    # Should synchronize coordinates since types match
    assert isinstance(data["crop_type"], str)
    assert array_equal(data["crop_coordinates"], array([10, 10, 90, 90]))


def test_migrate_both_coordinates_empty():
    """Test when both select_coordinates and crop_coordinates are empty."""
    input_data = {
        "selection_type": "rectangle",
        "select_coordinates": [],
        "crop_type": "rectangle",
        "crop_coordinates": [],
    }

    data = migrate_scratch_2_to_3_fields(input_data)

    # Both empty, should remain empty
    assert not data["crop_coordinates"]


def test_legacy_fields_removed_after_migration():
    """Test that legacy selection_type and select_coordinates are removed after migration."""
    test_cases = [
        # Case 1: Both exist with matching types
        {
            "selection_type": "rectangle",
            "select_coordinates": array([10, 10, 90, 90]),
            "crop_type": "rectangle",
            "crop_coordinates": array([0, 0, 100, 100]),
            "other_field": "preserved",
        },
        # Case 2: Both exist with different types
        {
            "selection_type": "circle",
            "select_coordinates": array([50, 50, 25]),
            "crop_type": "rectangle",
            "crop_coordinates": array([0, 0, 100, 100]),
            "other_field": "preserved",
        },
        # Case 3: Only selection exists
        {
            "selection_type": "polygon",
            "select_coordinates": array([0, 0, 100, 0, 100, 100]),
            "other_field": "preserved",
        },
    ]

    for input_data in test_cases:
        data = migrate_scratch_2_to_3_fields(input_data)

        # Legacy fields should not be in migrated data
        assert "selection_type" not in data, (
            f"Legacy 'selection_type' should be removed or empty. Got: {data}"
        )
        assert "select_coordinates" not in data, (
            f"Legacy 'select_coordinates' should be removed or empty. Got: {data}"
        )

        # Crop fields should be present
        assert "crop_type" in data, "crop_type should be present after migration"
        assert "crop_coordinates" in data, (
            "crop_coordinates should be present after migration"
        )

        # Other fields should be preserved
        assert data.get("other_field") == "preserved", (
            "Other fields should be preserved"
        )


def test_original_dict_not_mutated():
    """Test that the original dictionary is not modified during migration."""
    original_data = {
        "selection_type": "rectangle",
        "select_coordinates": array([10, 10, 90, 90]),
        "crop_type": "rectangle",
        "crop_coordinates": array([0, 0, 100, 100]),
    }

    # Create a snapshot of the original keys
    original_keys = set(original_data.keys())

    # Run migration
    migrated_data = migrate_scratch_2_to_3_fields(original_data)

    # Original data should still have the same keys
    assert not original_keys.difference(original_data.keys()), (
        "Original dictionary should not be mutated"
    )

    # Migrated data should be different
    assert migrated_data is not original_data, "Should return a new dictionary"
