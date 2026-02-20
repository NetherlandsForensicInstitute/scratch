from pathlib import Path

import numpy as np

from conversion.export.utils import check_if_file_exists
from conversion.profile_correlator import Profile


def save_profile(profile: Profile, path: Path) -> None:
    """
    Save a Profile object to a compressed NPZ file.

    Creates one file: {path}.npz containing heights and pixel_size.

    :param profile: Profile object to save
    :param path: File path (suffix is replaced for .npz output)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path.with_suffix(".npz"),
        heights=profile.heights,
        pixel_size=np.array(profile.pixel_size),
    )


def load_profile_from_path(path: Path, stem: str) -> Profile:
    """
    Load a Profile object from a compressed NPZ file.

    :param path: Directory path containing the file
    :param stem: Base filename
    :returns: Reconstructed Profile object
    :raises FileNotFoundError: If NPZ file does not exist
    """
    npz_file = (path / stem).with_suffix(".npz")
    check_if_file_exists(npz_file)
    data = np.load(npz_file)
    return Profile(heights=data["heights"], pixel_size=float(data["pixel_size"]))
