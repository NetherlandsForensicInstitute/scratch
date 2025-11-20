import re
from typing import Any, Literal

from scipy.constants import micro, milli, nano


def _extract_resolution_from_description(
    description_text: str, axis: Literal["Vertical", "Lateral"]
) -> float | None:
    """
    Parse the axis resolution value from description text.

    Parameters
    ----------
    description_text : str
        The full description text to search in
    axis : str
        The axis string to scrape

    Returns
    -------
    float | None
        Resolution value in meters, or None if parsing fails
    """
    if not (
        match_ := re.search(
            (
                f"Estimated {axis} Resolution:"
                r"\s*(?P<quantity>[\d.]+)\s*(?P<unit>[mnµ]+)"
            ),
            description_text,
        )
    ):
        return None

    try:
        value = float(match_.group("quantity"))
    except ValueError:
        return None

    # Extract unit and match against string value
    match match_.group("unit"):
        case "m":
            return value
        case "mm":
            return value * milli
        case "µm":
            return value * micro
        case "nm":
            return value * nano
        case _:
            return None


def extract_resolutions_from_xml_data(
    xml_data: dict[str, Any],
) -> tuple[float | None, float | None]:
    """
    Extract VR and LR from AL3D XMLData structure.
    VR: Vertical Resolution
    LR: Lateral Resolution

    Parameters
    ----------
    data : dict
        Data dictionary containing XMLData field (from AliconaReader)

    Returns
    -------
    tuple[float | None, float | None]
        (VR, LR) - Vertical and Lateral Resolution in meters, or None if not found

    """
    try:
        # data.XMLData.Object3D.generalData.description.Text
        description_text = xml_data["Object3D"]["generalData"]["description"]["Text"]
    except (KeyError, IndexError, ValueError, AttributeError):
        return None, None

    # Extract resolutions from description text
    return (
        _extract_resolution_from_description(description_text, "Lateral"),
        _extract_resolution_from_description(description_text, "Vertical"),
    )
