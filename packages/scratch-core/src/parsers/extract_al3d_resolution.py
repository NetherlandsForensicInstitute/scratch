import re
from typing import Any

from scipy.constants import micro, milli, nano


def _parse_resolution_value(resolution_str: str) -> float | None:
    """
    Parse resolution value and unit from string and convert to meters.

    This helper function extracts the numeric value and unit from a resolution string
    like "2.5 µm" and converts it to meters using the appropriate SI prefix.

    Parameters
    ----------
    resolution_str : str
        Resolution string containing value and unit (e.g., "2.5 µm", "500 nm")

    Returns
    -------
    float | None
        Resolution value in meters, or None if parsing fails
    """

    # Pattern matches: digits/decimal point (quantity), optional whitespace, unit characters
    if not (
        match_ := re.search(r"(?P<quantity>[\d.]+)\s*(?P<unit>[mnµ]+)", resolution_str)
    ):
        return None

    # Extract quantity
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
    data: dict[str, Any],
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
    VR = None
    LR = None

    # MATLAB: if isfield(data, 'XMLData')
    if "XMLData" not in data:
        # MATLAB: else block (lines 367-371)
        return None, None

    try:
        # MATLAB: data.XMLData.Object3D.generalData.description.Text
        description_text = data["XMLData"]["Object3D"]["generalData"]["description"][
            "Text"
        ]

        # MATLAB: strfind(..., 'Estimated Lateral Resolution:')
        # MATLAB: strfind(..., 'Estimated Vertical Resolution:')
        start_LR = description_text.find("Estimated Lateral Resolution:")
        start_VR = description_text.find("Estimated Vertical Resolution:")

        # ===== Extract Lateral Resolution =====
        # MATLAB: if ~isempty(start_LR)
        if start_LR != -1:
            # MATLAB: start_LR = start_LR + length('Estimated Lateral Resolution:')
            # Note: MATLAB uses 1-based indexing, Python uses 0-based
            start_LR = start_LR + len("Estimated Lateral Resolution:")

            # MATLAB: while uint8(text(start_LR)) == 32
            # Skip whitespace (ASCII 32)
            while (
                start_LR < len(description_text)
                and ord(description_text[start_LR]) == 32
            ):
                start_LR = start_LR + 1

            end_LR = start_LR

            # MATLAB: while uint8(text(end_LR)) ~= 13 && uint8(text(end_LR)) ~= 10
            # Find end of line (CR=13 or LF=10)
            while end_LR < len(description_text):
                char_code = ord(description_text[end_LR])
                if char_code == 13 or char_code == 10:  # CR or LF
                    break
                end_LR = end_LR + 1

            # MATLAB: end_LR = end_LR - 1
            # Note: In Python slicing, we don't need to subtract 1 because end is exclusive
            # But to match MATLAB's inclusive end, we use end_LR as-is in the slice

            # MATLAB: description.Estimated_Lateral_Resolution = Text(start_LR:end_LR)
            LR_string = description_text[start_LR:end_LR]
        else:
            LR_string = None

        # ===== Extract Vertical Resolution =====
        # MATLAB: if ~isempty(start_VR)
        if start_VR != -1:
            # Same process as LR
            start_VR = start_VR + len("Estimated Vertical Resolution:")

            while (
                start_VR < len(description_text)
                and ord(description_text[start_VR]) == 32
            ):
                start_VR = start_VR + 1

            end_VR = start_VR

            while end_VR < len(description_text):
                char_code = ord(description_text[end_VR])
                if char_code == 13 or char_code == 10:
                    break
                end_VR = end_VR + 1

            VR_string = description_text[start_VR:end_VR]
        else:
            VR_string = None

        # ===== Parse LR value and units =====
        if LR_string:
            LR = _parse_resolution_value(LR_string)

        # ===== Parse VR value and units =====
        if VR_string:
            VR = _parse_resolution_value(VR_string)

    except (KeyError, IndexError, ValueError, AttributeError):
        # MATLAB: catch block (lines 360-363)
        VR = None
        LR = None

    return VR, LR


# Example usage and test
if __name__ == "__main__":
    # Test with sample data
    test_data = {
        "XMLData": {
            "Object3D": {
                "generalData": {
                    "description": {
                        "Text": (
                            "Some metadata\n"
                            "Estimated Vertical Resolution: 2.5 µm\n"
                            "Estimated Lateral Resolution: 1.0 µm\n"
                            "More data"
                        )
                    }
                }
            }
        }
    }

    vr, lr = extract_resolutions_from_xml_data(test_data)
    print(f"VR: {vr} m ({vr * 1e6:.1f} µm)")
    print(f"LR: {lr} m ({lr * 1e6:.1f} µm)")

    # Test without XMLData
    empty_data = {}
    vr, lr = extract_resolutions_from_xml_data(empty_data)
    print(f"\nWithout XMLData - VR: {vr}, LR: {lr}")

    # Test with different units
    test_data_mm = {
        "XMLData": {
            "Object3D": {
                "generalData": {
                    "description": {"Text": "Estimated Vertical Resolution: 0.5 mm\n"}
                }
            }
        }
    }

    vr, lr = extract_resolutions_from_xml_data(test_data_mm)
    print(f"\nWith mm units - VR: {vr} m ({vr * 1e3:.1f} mm)")
