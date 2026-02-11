import json
from pathlib import PosixPath

import numpy as np
import pytest

from conversion.export.utils import (
    check_if_file_exists,
    load_compressed_binary,
    load_json,
    save_as_compressed_binary,
    save_as_json,
)


class TestCheckIfFileExists:
    """Tests for `check_if_file_exists` function."""

    def test_existing_file_passes(self, tmp_path: PosixPath):
        """Test that existing file doesn't raise exception."""
        test_file = tmp_path / "exists.txt"
        test_file.write_text("content")

        # Should not raise
        check_if_file_exists(test_file)

    def test_missing_file_raises_error(self, tmp_path: PosixPath):
        """Test that missing file raises FileNotFoundError."""
        missing_file = tmp_path / "does_not_exist.txt"

        with pytest.raises(
            FileNotFoundError, match='File ".*does_not_exist.txt" does not exist'
        ):
            check_if_file_exists(missing_file)

    def test_directory_raises_error(self, tmp_path: PosixPath):
        """Test that directory path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            check_if_file_exists(tmp_path)


class TestLoadJson:
    """Tests for `load_json` function."""

    def test_load_simple_json(self, tmp_path: PosixPath):
        """Test loading simple JSON data."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(data))

        loaded = load_json(json_file)

        assert loaded == data

    def test_load_nested_json(self, tmp_path: PosixPath):
        """Test loading nested JSON structure."""
        json_file = tmp_path / "nested.json"
        data = {"outer": {"inner": {"deep": "value"}}, "list": [1, 2, 3]}
        json_file.write_text(json.dumps(data))

        loaded = load_json(json_file)

        assert loaded == data

    def test_load_empty_json(self, tmp_path: PosixPath):
        """Test loading empty JSON object."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")

        loaded = load_json(json_file)

        assert loaded == {}

    def test_load_json_with_special_types(self, tmp_path: PosixPath):
        """Test loading JSON with various data types."""
        json_file = tmp_path / "types.json"
        data = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
        }
        json_file.write_text(json.dumps(data))

        loaded = load_json(json_file)

        assert loaded == data

    def test_load_invalid_json_raises_error(self, tmp_path: PosixPath):
        """Test that invalid JSON raises JSONDecodeError."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_json(json_file)

    def test_load_missing_file_raises_error(self, tmp_path: PosixPath):
        """Test that missing file raises FileNotFoundError."""
        missing_file = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_json(missing_file)


class TestSaveAsJson:
    """Tests for `save_as_json` function."""

    def test_save_simple_json(self, tmp_path: PosixPath):
        """Test saving simple JSON string."""
        file_path = tmp_path / "test"
        data = json.dumps({"key": "value"})

        save_as_json(data, file_path)

        assert (tmp_path / "test.json").exists()
        with open(tmp_path / "test.json", "r") as f:
            loaded = json.load(f)
        assert loaded == {"key": "value"}

    def test_save_overwrites_existing(self, tmp_path: PosixPath):
        """Test that saving overwrites existing file."""
        file_path = tmp_path / "test"

        save_as_json(json.dumps({"first": "data"}), file_path)
        save_as_json(json.dumps({"second": "data"}), file_path)

        with open(tmp_path / "test.json", "r") as f:
            loaded = json.load(f)
        assert loaded == {"second": "data"}

    def test_save_changes_extension_to_json(self, tmp_path: PosixPath):
        """Test that .json extension is always used."""
        file_path = tmp_path / "test.txt"

        save_as_json(json.dumps({"key": "value"}), file_path)

        assert (tmp_path / "test.json").exists()
        assert not (tmp_path / "test.txt").exists()

    def test_save_formatted_json(self, tmp_path: PosixPath):
        """Test saving formatted JSON with indentation."""
        file_path = tmp_path / "formatted"
        data = json.dumps({"key": "value", "nested": {"inner": "data"}}, indent=4)

        save_as_json(data, file_path)

        content = (tmp_path / "formatted.json").read_text()
        assert "\n" in content  # Formatted JSON should have newlines


class TestLoadCompressedBinary:
    """Tests for `load_compressed_binary` function."""

    def test_load_1d_array(self, tmp_path: PosixPath):
        """Test loading 1D numpy array."""
        npz_file = tmp_path / "test.npz"
        original = np.array([1, 2, 3, 4, 5])
        np.savez_compressed(npz_file, data=original)

        loaded = load_compressed_binary(npz_file)

        np.testing.assert_array_equal(loaded, original)

    def test_load_2d_array(self, tmp_path: PosixPath):
        """Test loading 2D numpy array."""
        npz_file = tmp_path / "test.npz"
        original = np.array([[1, 2, 3], [4, 5, 6]])
        np.savez_compressed(npz_file, data=original)

        loaded = load_compressed_binary(npz_file)

        np.testing.assert_array_equal(loaded, original)

    def test_load_float_array(self, tmp_path: PosixPath):
        """Test loading array with float values."""
        npz_file = tmp_path / "test.npz"
        original = np.array([1.5, 2.7, 3.14, 42.0])
        np.savez_compressed(npz_file, data=original)

        loaded = load_compressed_binary(npz_file)

        np.testing.assert_array_equal(loaded, original)

    def test_load_preserves_dtype(self, tmp_path: PosixPath):
        """Test that loading preserves array dtype."""
        npz_file = tmp_path / "test.npz"
        original = np.array([1, 2, 3], dtype=np.int32)
        np.savez_compressed(npz_file, data=original)

        loaded = load_compressed_binary(npz_file)

        assert loaded.dtype == np.int32

    @pytest.mark.integration
    def test_load_large_array(self, tmp_path: PosixPath):
        """Test loading large array to verify compression works."""
        npz_file = tmp_path / "large.npz"
        original = np.random.rand(1000, 1000)
        np.savez_compressed(npz_file, data=original)

        loaded = load_compressed_binary(npz_file)

        np.testing.assert_array_equal(loaded, original)

    def test_load_missing_file_raises_error(self, tmp_path: PosixPath):
        """Test that missing file raises FileNotFoundError."""
        missing_file = tmp_path / "missing.npz"

        with pytest.raises(FileNotFoundError):
            load_compressed_binary(missing_file)

    def test_load_missing_data_key_raises_error(self, tmp_path: PosixPath):
        """Test that NPZ without 'data' key raises KeyError."""
        npz_file = tmp_path / "wrong_key.npz"
        np.savez_compressed(npz_file, wrong_key=np.array([1, 2, 3]))

        with pytest.raises(KeyError):
            load_compressed_binary(npz_file)


class TestSaveAsCompressedBinary:
    """Tests for `save_as_compressed_binary` function."""

    def test_save_1d_array(self, tmp_path: PosixPath):
        """Test saving 1D numpy array."""
        file_path = tmp_path / "test"
        array = np.array([1, 2, 3, 4, 5])

        save_as_compressed_binary(array, file_path)

        assert (tmp_path / "test.npz").exists()
        with np.load(tmp_path / "test.npz") as loaded:
            np.testing.assert_array_equal(loaded["data"], array)

    def test_save_2d_array(self, tmp_path: PosixPath):
        """Test saving 2D numpy array."""
        file_path = tmp_path / "test"
        array = np.array([[1, 2], [3, 4], [5, 6]])

        save_as_compressed_binary(array, file_path)

        with np.load(tmp_path / "test.npz") as loaded:
            np.testing.assert_array_equal(loaded["data"], array)

    def test_save_overwrites_existing(self, tmp_path: PosixPath):
        """Test that saving overwrites existing file."""
        file_path = tmp_path / "test"

        save_as_compressed_binary(np.array([1, 2, 3]), file_path)
        save_as_compressed_binary(np.array([4, 5, 6]), file_path)

        with np.load(tmp_path / "test.npz") as loaded:
            np.testing.assert_array_equal(loaded["data"], np.array([4, 5, 6]))

    def test_save_changes_extension_to_npz(self, tmp_path: PosixPath):
        """Test that .npz extension is always used."""
        file_path = tmp_path / "test.dat"

        save_as_compressed_binary(np.array([1, 2, 3]), file_path)

        assert (tmp_path / "test.npz").exists()
        assert not (tmp_path / "test.dat").exists()

    def test_save_preserves_dtype(self, tmp_path: PosixPath):
        """Test that saving preserves array dtype."""
        file_path = tmp_path / "test"
        array = np.array([1, 2, 3], dtype=np.float64)

        save_as_compressed_binary(array, file_path)

        with np.load(tmp_path / "test.npz") as loaded:
            assert loaded["data"].dtype == np.float64

    def test_save_empty_array(self, tmp_path: PosixPath):
        """Test saving empty array."""
        file_path = tmp_path / "empty"
        array = np.array([])

        save_as_compressed_binary(array, file_path)

        with np.load(tmp_path / "empty.npz") as loaded:
            np.testing.assert_array_equal(loaded["data"], array)


class TestRoundtrip:
    """Tests for save/load roundtrip operations."""

    def test_json_roundtrip(self, tmp_path: PosixPath):
        """Test complete JSON save/load cycle."""
        file_path = tmp_path / "test"
        original_data = {
            "string": "value",
            "number": 42,
            "nested": {"key": "value"},
        }

        save_as_json(json.dumps(original_data), file_path)
        loaded_data = load_json(tmp_path / "test.json")

        assert loaded_data == original_data

    def test_binary_roundtrip(self, tmp_path: PosixPath):
        """Test complete binary save/load cycle."""
        file_path = tmp_path / "test"
        original_array = np.random.rand(100, 100)

        save_as_compressed_binary(original_array, file_path)
        loaded_array = load_compressed_binary(tmp_path / "test.npz")

        np.testing.assert_array_equal(loaded_array, original_array)

    def test_multiple_roundtrips(self, tmp_path: PosixPath):
        """Test multiple save/load cycles maintain data integrity."""
        file_path = tmp_path / "test"
        original_array = np.array([[1.5, 2.5], [3.5, 4.5]])

        # First roundtrip
        save_as_compressed_binary(original_array, file_path)
        loaded1 = load_compressed_binary(tmp_path / "test.npz")

        # Second roundtrip
        save_as_compressed_binary(loaded1, file_path)
        loaded2 = load_compressed_binary(tmp_path / "test.npz")

        np.testing.assert_array_equal(loaded2, original_array)
