from pathlib import Path

from .base import parse_image, parse_scan
from .data_types import ParsedImage, ImageFileFormats, ScanFileFormats
from .x3p import save_to_x3p

__all__ = ("save_to_x3p", "parse_image", "parse_scan", "parse_surface_scan_file")

FILETYPE_TO_PARSER = {f".{ext}": parse_image for ext in ImageFileFormats} | {
    f".{ext}": parse_scan for ext in ScanFileFormats
}


def parse_surface_scan_file(path_to_scan_file: Path) -> ParsedImage:
    """
    Parse a surface scan file return an instance of `ParsedImage`.

    :param path_to_scan_file: Path to the surface scan file.
    :returns: An instance of `ParsedImage` containing the parsed scan data.
    """
    ext = path_to_scan_file.suffix.lower()
    try:
        parser = FILETYPE_TO_PARSER[ext]
    except KeyError:
        raise ValueError(f"File type not supported: {ext}")

    return parser(path_to_scan_file)
