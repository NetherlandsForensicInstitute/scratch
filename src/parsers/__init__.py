from pathlib import Path

from .base import parse_image, parse_scan
from .data_types import ParsedImage
from .x3p import parse_x3p, save_to_x3p

__all__ = ("save_to_x3p", "parse_image", "parse_scan", "parse_x3p")

FILETYPE_TO_PARSER = {
    ".png": parse_image,
    ".al3d": parse_scan,
    ".x3p": parse_x3p,
    # TODO: add more file types
}


def parse_file(path: Path) -> ParsedImage:
    """Parse a surface scan file and return an instance of `ParsedImage`."""
    parser = FILETYPE_TO_PARSER.get(path.suffix.lower())
    if parser:
        return parser(path)
    else:
        raise RuntimeError(f"File type not supported: {path.suffix}")
