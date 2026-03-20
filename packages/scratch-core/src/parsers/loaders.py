from functools import lru_cache
from pathlib import Path

from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from .patches.al3d import read_al3d

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


@lru_cache(maxsize=1)
def _load_surface(scan_file: Path) -> Surface:
    """
    Cache scan file as a Surface.
    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `Surface`.
    """
    return Surface.load(scan_file)
