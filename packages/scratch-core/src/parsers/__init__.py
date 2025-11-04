from pathlib import Path

from .base import parse_image, parse_scan
from .data_types import ParsedImage
from .x3p import save_to_x3p

__all__ = ("save_to_x3p", "parse_image", "parse_scan")

FILETYPE_TO_PARSER = {
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".bmp": parse_image,
    ".tif": parse_image,
    ".tiff": parse_image,
    ".al3d": parse_scan,
    ".x3p": parse_scan,
    ".sur": parse_scan,
    ".plu": parse_scan,
}


def parse_file(path: Path) -> ParsedImage:
    """Parse a surface scan file and return an instance of `ParsedImage`."""
    parser = FILETYPE_TO_PARSER.get(path.suffix.lower())
    if parser:
        return parser(path)
    else:
        raise RuntimeError(f"File type not supported: {path.suffix}")
