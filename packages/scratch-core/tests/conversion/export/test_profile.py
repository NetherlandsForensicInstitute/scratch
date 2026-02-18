from pathlib import PosixPath

import numpy as np
import pytest

from conversion.export.profile import load_profile_from_path, save_profile
from conversion.profile_correlator import Profile


@pytest.mark.integration
class TestSaveAndLoadProfile:
    """Tests for `save_profile` and `load_profile_from_path` functions."""

    def test_roundtrip(self, tmp_path: PosixPath, profile_with_nans: Profile):
        """Test that save/load roundtrip preserves all data."""
        save_profile(profile_with_nans, tmp_path, "test_profile")
        loaded = load_profile_from_path(tmp_path, "test_profile")

        assert np.isclose(loaded.pixel_size, profile_with_nans.pixel_size)
        np.testing.assert_array_equal(loaded.heights, profile_with_nans.heights)

    def test_load_missing_file(self, tmp_path: PosixPath):
        """Test that loading raises FileNotFoundError when NPZ is missing."""
        with pytest.raises(
            FileNotFoundError, match='File ".*test_profile.npz" does not exist'
        ):
            load_profile_from_path(tmp_path, "test_profile")
