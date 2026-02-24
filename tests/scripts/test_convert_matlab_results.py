"""Tests for convert_matlab_results.py."""

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.convert_matlab_results import (
    ConversionConfig,
    _data_param_field,
    _local_tag,
    _parse_circle,
    _parse_ellipse,
    _parse_rectangle,
    _run_parallel,
    _scalar,
    _x3p_metadata,
    convert_mark,
    convert_measurement_x3p,
    convert_x3p,
    copy_db_scratch_files,
    extract_impression_params,
    extract_mark_type,
    extract_mask_and_bounding_box,
    extract_striation_params,
    find_mark_folders,
    get_x3p_shape,
    main,
)


def make_x3p(  # noqa: PLR0913
    tmp_path: Path,
    name: str,
    nx: int,
    ny: int,
    n_layers: int = 1,
    dtype: str = "D",
) -> Path:
    """Create a minimal x3p zip file for testing.

    :param tmp_path: Directory to create the file in.
    :param name: Filename for the x3p archive.
    :param nx: Number of points in X.
    :param ny: Number of points in Y.
    :param n_layers: Number of data layers (default 1).
    :param dtype: Z-axis data type letter (``D``, ``F``, ``I``, or ``L``).
    :return: Path to the created x3p file.
    """
    xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<ISO5436_2 xmlns="http://www.opengps.eu/2008/ISO5436_2">
  <Record1>
    <Axes>
      <CX><AxisType>I</AxisType><DataType>D</DataType><Increment>1e-6</Increment><Offset>0</Offset></CX>
      <CY><AxisType>I</AxisType><DataType>D</DataType><Increment>1e-6</Increment><Offset>0</Offset></CY>
      <CZ><AxisType>A</AxisType><DataType>{dtype}</DataType><Increment>1</Increment><Offset>0</Offset></CZ>
    </Axes>
  </Record1>
  <Record3>
    <MatrixDimension>
      <SizeX>{nx}</SizeX>
      <SizeY>{ny}</SizeY>
      <SizeZ>1</SizeZ>
    </MatrixDimension>
    <DataLink>
      <PointDataLink>bindata/data.bin</PointDataLink>
      <MD5ChecksumPointData>placeholder</MD5ChecksumPointData>
    </DataLink>
  </Record3>
  <Record4>
    <ChecksumFile>md5checksum.hex</ChecksumFile>
  </Record4>
</ISO5436_2>"""

    np_dtype = {"D": np.float64, "F": np.float32, "I": np.int16, "L": np.int32}[dtype]
    layers = []
    for i in range(n_layers):
        layer = np.arange(nx * ny, dtype=np_dtype) + i * 1000
        layers.append(layer)
    data_bin = np.concatenate(layers).tobytes()
    data_md5 = hashlib.md5(data_bin).hexdigest()  # noqa: S324 — md5 required by x3p format

    xml = xml.replace("placeholder", data_md5)
    xml_bytes = xml.encode("utf-8")
    xml_md5 = hashlib.md5(xml_bytes).hexdigest()  # noqa: S324

    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("main.xml", xml_bytes)
        zf.writestr("bindata/data.bin", data_bin)
        zf.writestr("md5checksum.hex", xml_md5)

    return path


def make_matlab_struct(crop_type="ellipse", crop_params=None, mark_type="Firing pin impression mark"):
    """Build a mock MATLAB struct as a numpy structured array.

    :param crop_type: Crop shape type (``ellipse``, ``circle``, or ``rectangle``).
    :param crop_params: Crop parameters tuple; defaults per crop type if ``None``.
    :param mark_type: Mark type string stored in the struct.
    :return: Numpy structured array mimicking a loaded MATLAB struct.
    """
    if crop_params is None:
        if crop_type == "ellipse":
            crop_params = (np.array([100.0, 80.0]), 50.0, 60.0, 0.0)
        elif crop_type == "circle":
            crop_params = (np.array([100.0, 80.0]), 50.0)
        elif crop_type == "rectangle":
            crop_params = (np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]]),)

    crop_info = np.array([crop_type, crop_params, 1], dtype=object)

    dp_dtype = np.dtype([
        ("bAdjustPixelSpacing", object),
        ("bLevelOffset", object),
        ("bLevelTilt", object),
        ("bLevel2nd", object),
        ("intMeth", object),
        ("cut_borders_after_smoothing", object),
        ("use_mean", object),
        ("angle_accuracy", object),
    ])
    dp = np.array(
        [(1, 1, 1, 1, "cubic", 1, 1, 0.1)],
        dtype=dp_dtype,
    )

    struct_dtype = np.dtype([
        ("mark_type", object),
        ("crop_info", object),
        ("cutoff_hi", object),
        ("cutoff_lo", object),
        ("subsampling", object),
        ("data_param", object),
    ])
    struct = np.array(
        [(mark_type, crop_info, np.float64(250), np.float64(5), np.float64(1), dp)],
        dtype=struct_dtype,
    )
    return struct


class TestScalar:
    """Tests for :func:`_scalar`."""

    def test_0d_array(self):
        """Unwrap a 0-d numpy array to its Python scalar."""
        assert _scalar(np.array(7.5)) == 7.5  # noqa: PLR2004

    def test_nested_0d(self):
        """Unwrap a doubly-wrapped 0-d array."""
        assert _scalar(np.array(np.array(3.14))) == 3.14  # noqa: PLR2004

    def test_single_element_array(self):
        """Extract the value from a single-element 1-d array."""
        assert _scalar(np.array([42])) == 42  # noqa: PLR2004

    def test_multi_element_unchanged(self):
        """Return multi-element arrays as-is."""
        arr = np.array([1, 2, 3])
        assert _scalar(arr) is arr

    def test_plain_value(self):
        """Pass through non-array values unchanged."""
        assert _scalar("test") == "test"

    def test_empty_array_returns_none(self):
        """Return ``None`` for empty arrays."""
        assert _scalar(np.array([])) is None
        assert _scalar(np.array([], dtype=np.uint8)) is None


class TestDataParamField:
    """Tests for :func:`_data_param_field`."""

    def test_missing_field_returns_default(self):
        """Return the default when the requested field is not in the sub-struct."""
        struct = make_matlab_struct()
        assert _data_param_field(struct, "nonexistent_field", default="fallback") == "fallback"

    def test_missing_field_returns_none_by_default(self):
        """Return ``None`` when the field is absent and no default is given."""
        struct = make_matlab_struct()
        assert _data_param_field(struct, "nonexistent_field") is None


class TestLocalTag:
    """Tests for :func:`_local_tag`."""

    def test_with_namespace(self):
        """Strip namespace prefix from a qualified element tag."""
        el = ET.Element("{http://example.com}SizeX")
        assert _local_tag(el) == "SizeX"

    def test_without_namespace(self):
        """Return unqualified tag unchanged."""
        el = ET.Element("SizeX")
        assert _local_tag(el) == "SizeX"


class TestX3pMetadata:
    """Tests for :func:`_x3p_metadata`."""

    def test_dimensions_with_namespace(self):
        """Parse SizeX/SizeY from namespaced XML."""
        xml = """<root xmlns="http://www.opengps.eu/2008/ISO5436_2">
            <Record3><MatrixDimension>
                <SizeX>100</SizeX><SizeY>200</SizeY>
            </MatrixDimension></Record3>
        </root>"""
        root = ET.fromstring(xml)  # noqa: S314 — trusted test data
        nx, ny, _ = _x3p_metadata(root)
        assert (nx, ny) == (100, 200)

    def test_dimensions_without_namespace(self):
        """Parse SizeX/SizeY from plain XML without namespace."""
        xml = "<root><Record3><MatrixDimension><SizeX>50</SizeX><SizeY>60</SizeY></MatrixDimension></Record3></root>"
        root = ET.fromstring(xml)  # noqa: S314
        nx, ny, _ = _x3p_metadata(root)
        assert (nx, ny) == (50, 60)

    def test_missing_dimensions_returns_zeros(self):
        """Return ``(0, 0)`` when MatrixDimension is absent."""
        root = ET.fromstring("<root></root>")  # noqa: S314
        nx, ny, _ = _x3p_metadata(root)
        assert (nx, ny) == (0, 0)

    @pytest.mark.parametrize(
        ("letter", "expected"),
        [("D", np.float64), ("F", np.float32), ("I", np.int16), ("L", np.int32)],
    )
    def test_known_dtypes(self, letter, expected):
        """Map CZ DataType letters to correct numpy dtypes."""
        xml = f"<root><CZ><DataType>{letter}</DataType></CZ></root>"
        root = ET.fromstring(xml)  # noqa: S314
        _, _, z_dtype = _x3p_metadata(root)
        assert z_dtype == np.dtype(expected)

    def test_missing_dtype_defaults_float32(self):
        """Default to float32 when CZ DataType is absent."""
        root = ET.fromstring("<root></root>")  # noqa: S314
        _, _, z_dtype = _x3p_metadata(root)
        assert z_dtype == np.dtype(np.float32)

    def test_full_metadata(self):
        """Parse both dimensions and dtype from a complete XML."""
        xml = """<root>
            <Record3><MatrixDimension><SizeX>10</SizeX><SizeY>20</SizeY></MatrixDimension></Record3>
            <CZ><DataType>D</DataType></CZ>
        </root>"""
        root = ET.fromstring(xml)  # noqa: S314
        nx, ny, z_dtype = _x3p_metadata(root)
        assert (nx, ny) == (10, 20)
        assert z_dtype == np.dtype(np.float64)


class TestConvertX3p:
    """Tests for :func:`convert_x3p`."""

    def test_single_layer_no_swap(self, tmp_path):
        """Single-layer x3p data is copied unchanged."""
        src = make_x3p(tmp_path, "in.x3p", nx=4, ny=3, n_layers=1)
        dst = tmp_path / "out.x3p"
        convert_x3p(src, dst)

        assert dst.exists()

        with zipfile.ZipFile(src) as z_in, zipfile.ZipFile(dst) as z_out:
            orig = np.frombuffer(z_in.read("bindata/data.bin"), dtype=np.float64)
            conv = np.frombuffer(z_out.read("bindata/data.bin"), dtype=np.float64)
        np.testing.assert_array_equal(orig, conv)

    def test_two_layers_swapped(self, tmp_path):
        """Two-layer x3p has its layers swapped in the output."""
        src = make_x3p(tmp_path, "in.x3p", nx=4, ny=3, n_layers=2)
        dst = tmp_path / "out.x3p"
        convert_x3p(src, dst)

        with zipfile.ZipFile(src) as z_in, zipfile.ZipFile(dst) as z_out:
            orig = np.frombuffer(z_in.read("bindata/data.bin"), dtype=np.float64)
            conv = np.frombuffer(z_out.read("bindata/data.bin"), dtype=np.float64)

        n = 4 * 3
        np.testing.assert_array_equal(conv[:n], orig[n:])
        np.testing.assert_array_equal(conv[n:], orig[:n])

    def test_checksums_updated(self, tmp_path):
        """MD5 checksums in the output match the actual data and XML."""
        src = make_x3p(tmp_path, "in.x3p", nx=4, ny=3, n_layers=2)
        dst = tmp_path / "out.x3p"
        convert_x3p(src, dst)

        with zipfile.ZipFile(dst) as z:
            data_bin = z.read("bindata/data.bin")
            xml_bytes = z.read("main.xml")
            stored_xml_md5 = z.read("md5checksum.hex").decode()

            root = ET.parse(BytesIO(xml_bytes)).getroot()  # noqa: S314
            ns = {"ns": "http://www.opengps.eu/2008/ISO5436_2"}
            stored_data_md5 = root.find(".//ns:MD5ChecksumPointData", ns)

        if stored_data_md5:
            stored_data_md5 = stored_data_md5.text
            assert stored_data_md5 == hashlib.md5(data_bin).hexdigest()  # noqa: S324
            assert stored_xml_md5 == hashlib.md5(xml_bytes).hexdigest()  # noqa: S324

    def test_raises_no_data_bin(self, tmp_path):
        """Raise :class:`ValueError` when ``data.bin`` is missing from the archive."""
        path = tmp_path / "bad.x3p"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "main.xml",
                "<root><Record3><MatrixDimension><SizeX>1</SizeX><SizeY>1</SizeY></MatrixDimension></Record3><CZ><DataType>D</DataType></CZ></root>",
            )
        with pytest.raises(ValueError, match="No data.bin"):
            convert_x3p(path, tmp_path / "out.x3p")

    def test_raises_zero_dimensions(self, tmp_path):
        """Raise :class:`ValueError` when SizeX or SizeY is zero."""
        xml = "<root><Record3><MatrixDimension><SizeX>0</SizeX><SizeY>5</SizeY></MatrixDimension></Record3></root>"
        path = tmp_path / "bad.x3p"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("main.xml", xml)
            zf.writestr("bindata/data.bin", b"")
        with pytest.raises(ValueError, match="Could not determine dimensions"):
            convert_x3p(path, tmp_path / "out.x3p")


class TestGetX3pShape:
    """Tests for :func:`get_x3p_shape`."""

    def test_reads_dimensions(self, tmp_path):
        """Return ``(nx, ny)`` from a valid x3p file."""
        x3p = make_x3p(tmp_path, "test.x3p", nx=100, ny=200)
        assert get_x3p_shape(x3p) == (100, 200)

    def test_raises_on_missing_dimensions(self, tmp_path):
        """Raise :class:`ValueError` when SizeX/SizeY elements are missing."""
        path = tmp_path / "bad.x3p"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("main.xml", "<root><MatrixDimension></MatrixDimension></root>")
        with pytest.raises(ValueError, match="Missing SizeX/SizeY"):
            get_x3p_shape(path)


class TestParseEllipse:
    """Tests for :func:`_parse_ellipse`."""

    def test_from_tuple(self):
        """Parse center, axes, and angle from a plain tuple."""
        raw = (np.array([100.0, 80.0]), 50.0, 60.0, 30.0)
        result = _parse_ellipse(raw)
        np.testing.assert_array_equal(result["center"], [100.0, 80.0])
        assert result["minor"] == 50.0  # noqa: PLR2004
        assert result["major"] == 60.0  # noqa: PLR2004
        assert result["angle"] == 30.0  # noqa: PLR2004

    def test_from_0d_wrapped_tuple(self):
        """Parse ellipse from a 0-d object array wrapping a tuple."""
        raw = np.array((np.array([100.0, 80.0]), 50.0, 60.0, 30.0), dtype=object)
        result = _parse_ellipse(raw)
        np.testing.assert_array_equal(result["center"], [100.0, 80.0])


class TestParseCircle:
    """Tests for :func:`_parse_circle`."""

    def test_from_tuple(self):
        """Parse circle as an ellipse with equal axes and zero angle."""
        raw = (np.array([50.0, 60.0]), 25.0)
        result = _parse_circle(raw)
        np.testing.assert_array_equal(result["center"], [50.0, 60.0])
        assert result["minor"] == 25.0  # noqa: PLR2004
        assert result["major"] == 25.0  # noqa: PLR2004
        assert result["angle"] == 0.0


class TestParseRectangle:
    """Tests for :func:`_parse_rectangle`."""

    def test_from_tuple(self):
        """Extract the corner array from a single-element tuple."""
        corners = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]])
        raw = (corners,)
        result = _parse_rectangle(raw)
        np.testing.assert_array_equal(result, corners)


class TestExtractMaskAndBoundingBox:
    """Tests for :func:`extract_mask_and_bounding_box`."""

    def test_ellipse_mask_shape_and_flip(self):
        """Ellipse produces a boolean mask of correct shape with no bounding box."""
        struct = make_matlab_struct("ellipse", (np.array([100.0, 80.0]), 50.0, 60.0, 0.0))
        mask, bbox = extract_mask_and_bounding_box(struct, 200, 160)

        assert mask.shape == (160, 200)
        assert mask.dtype == bool
        assert bbox is None
        assert mask.any()

    def test_ellipse_mask_is_vertically_flipped(self):
        """Mask is flipped vertically to convert from MATLAB y-up convention."""
        struct = make_matlab_struct("ellipse", (np.array([100.0, 40.0]), 20.0, 20.0, 0.0))
        mask, _ = extract_mask_and_bounding_box(struct, 200, 160)

        rows_with_true = np.where(mask.any(axis=1))[0]
        center_row = rows_with_true.mean()
        assert center_row > 80, f"Expected center near bottom, got row {center_row}"  # noqa: PLR2004

    def test_circle_mask(self):
        """Circle produces a boolean mask with no bounding box."""
        struct = make_matlab_struct("circle", (np.array([50.0, 50.0]), 20.0))
        mask, bbox = extract_mask_and_bounding_box(struct, 100, 100)

        assert mask.shape == (100, 100)
        assert bbox is None
        assert mask.any()

    def test_rectangle_mask_and_bbox(self):
        """Rectangle produces both a mask and a flipped bounding box."""
        corners = np.array([[10.0, 20.0], [90.0, 20.0], [90.0, 80.0], [10.0, 80.0]])
        struct = make_matlab_struct("rectangle", (corners,))
        mask, bbox_list = extract_mask_and_bounding_box(struct, 100, 100)

        assert mask.shape == (100, 100)
        assert bbox_list is not None
        assert mask.any()

    def test_unknown_crop_type_raises(self):
        """Raise :class:`ValueError` for unsupported crop types."""
        struct = make_matlab_struct("polygon", (np.array([1, 2]),))
        with pytest.raises(ValueError, match="Unknown crop type"):
            extract_mask_and_bounding_box(struct, 100, 100)


class TestExtractParams:
    """Tests for :func:`extract_impression_params` and :func:`extract_striation_params`."""

    def test_impression_params(self):
        """Extract impression mark parameters with cutoffs converted to metres."""
        struct = make_matlab_struct(mark_type="Firing pin impression mark")
        mark_type = extract_mark_type(struct)
        params = extract_impression_params(struct, mark_type)

        assert params["adjust_pixel_spacing"] is True
        assert params["level_offset"] is True
        assert params["interp_method"] == "cubic"
        assert params["highpass_cutoff"] == pytest.approx(250e-6)
        assert params["lowpass_cutoff"] == pytest.approx(5e-6)

    def test_impression_params_empty_cutoffs(self):
        """Return ``None`` cutoffs when the MATLAB fields are empty arrays."""
        struct = make_matlab_struct(mark_type="Firing pin impression mark")
        struct["cutoff_hi"][0] = np.array([], dtype=np.uint8)
        struct["cutoff_lo"][0] = np.array([], dtype=np.uint8)
        mark_type = extract_mark_type(struct)
        params = extract_impression_params(struct, mark_type)

        assert params["highpass_cutoff"] is None
        assert params["lowpass_cutoff"] is None

    def test_striation_params(self):
        """Extract striation mark parameters including angle accuracy."""
        struct = make_matlab_struct(mark_type="Striation mark")
        params = extract_striation_params(struct)

        assert params["highpass_cutoff"] == pytest.approx(250e-6)
        assert params["lowpass_cutoff"] == pytest.approx(5e-6)
        assert params["cut_borders_after_smoothing"] is True
        assert params["use_mean"] is True
        assert params["angle_accuracy"] == pytest.approx(0.1)
        assert params["subsampling_factor"] == 1

    def test_striation_params_defaults(self):
        """Fall back to default cutoffs when MATLAB fields are empty."""
        struct = make_matlab_struct(mark_type="Striation mark")
        struct["cutoff_hi"][0] = np.array([], dtype=np.uint8)
        struct["cutoff_lo"][0] = np.array([], dtype=np.uint8)
        params = extract_striation_params(struct)

        assert params["highpass_cutoff"] == pytest.approx(2e-3)
        assert params["lowpass_cutoff"] == pytest.approx(2.5e-4)


class TestFindMarkFolders:
    """Tests for :func:`find_mark_folders`."""

    def test_finds_marks(self, tmp_path):
        """Yield ``(measurement_dir, mark_dir)`` for valid mark folders."""
        meas = tmp_path / "tool" / "rep" / "scan"
        mark = meas / "firing_pin"
        mark.mkdir(parents=True)
        (mark / "mark.mat").touch()
        make_x3p(meas, "measurement.x3p", nx=2, ny=2)

        results = list(find_mark_folders(tmp_path))
        assert len(results) == 1
        assert results[0] == (meas, mark)

    def test_skips_without_x3p(self, tmp_path):
        """Skip mark folders whose parent has no measurement x3p."""
        mark = tmp_path / "tool" / "rep" / "scan" / "firing_pin"
        mark.mkdir(parents=True)
        (mark / "mark.mat").touch()

        results = list(find_mark_folders(tmp_path))
        assert len(results) == 0

    def test_multiple_marks_same_measurement(self, tmp_path):
        """Find multiple mark folders under the same measurement directory."""
        meas = tmp_path / "tool" / "rep" / "scan"
        mark1 = meas / "firing_pin"
        mark2 = meas / "breech_face"
        mark1.mkdir(parents=True)
        mark2.mkdir(parents=True)
        (mark1 / "mark.mat").touch()
        (mark2 / "mark.mat").touch()
        make_x3p(meas, "measurement.x3p", nx=2, ny=2)

        results = list(find_mark_folders(tmp_path))
        assert len(results) == 2  # noqa: PLR2004


class TestCopyDbScratchFiles:
    """Tests for :func:`copy_db_scratch_files`."""

    def test_copies_with_timestamp(self, tmp_path):
        """Copy db.scratch files and append a ``CONVERTED_DATE`` line."""
        root = tmp_path / "root"
        out = tmp_path / "out"
        (root / "a").mkdir(parents=True)
        (root / "a" / "db.scratch").write_text("KEY=value\n")

        count = copy_db_scratch_files(root, out)

        assert count == 1
        content = (out / "a" / "db.scratch").read_text()
        assert "KEY=value" in content
        assert "CONVERTED_DATE=" in content

    def test_overwrites_existing(self, tmp_path):
        """Overwrite a pre-existing db.scratch in the output directory."""
        root = tmp_path / "root"
        out = tmp_path / "out"
        (root / "a").mkdir(parents=True)
        (root / "a" / "db.scratch").write_text("KEY=value\n")
        (out / "a").mkdir(parents=True)
        (out / "a" / "db.scratch").write_text("ALREADY_HERE\n")

        copy_db_scratch_files(root, out)

        content = (out / "a" / "db.scratch").read_text()
        assert "KEY=value" in content
        assert "CONVERTED_DATE=" in content
        assert "ALREADY_HERE" not in content


class TestConversionConfig:
    """Tests for :class:`ConversionConfig`."""

    def test_strips_trailing_slash(self):
        """``__post_init__`` strips the trailing slash from ``api_url``."""
        cfg = ConversionConfig(root=Path("/r"), output_dir=Path("/o"), api_url="http://localhost:8000/")
        assert cfg.api_url == "http://localhost:8000"

    def test_leaves_clean_url(self):
        """URLs without a trailing slash are left unchanged."""
        cfg = ConversionConfig(root=Path("/r"), output_dir=Path("/o"), api_url="http://localhost:8000")
        assert cfg.api_url == "http://localhost:8000"


class TestConvertMeasurementX3p:
    """Tests for :func:`convert_measurement_x3p`."""

    def test_skips_existing(self, tmp_path):
        """Skip conversion when the output file already exists."""
        root = tmp_path / "root" / "meas"
        root.mkdir(parents=True)
        make_x3p(root, "measurement.x3p", nx=2, ny=2)

        out = tmp_path / "out" / "meas"
        out.mkdir(parents=True)
        (out / "measurement.x3p").write_text("already here")

        cfg = ConversionConfig(root=tmp_path / "root", output_dir=tmp_path / "out", api_url="http://x")

        result = convert_measurement_x3p(root, cfg)
        assert result == out / "measurement.x3p"
        assert (out / "measurement.x3p").read_text() == "already here"

    def test_force_overrides_skip(self, tmp_path):
        """Overwrite existing output when ``force=True``."""
        root = tmp_path / "root" / "meas"
        root.mkdir(parents=True)
        make_x3p(root, "measurement.x3p", nx=2, ny=2)

        out = tmp_path / "out" / "meas"
        out.mkdir(parents=True)
        (out / "measurement.x3p").write_text("old")

        cfg = ConversionConfig(
            root=tmp_path / "root",
            output_dir=tmp_path / "out",
            api_url="http://x",
            force=True,
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        with patch("scripts.convert_matlab_results.requests.post", return_value=mock_resp):
            result = convert_measurement_x3p(root, cfg)

        assert result == out / "measurement.x3p"
        assert (out / "measurement.x3p").read_bytes() != b"old"

    def test_downloads_preview_and_surface_map(self, tmp_path):
        """Download preview and surface_map images returned by the API."""
        root = tmp_path / "root" / "meas"
        root.mkdir(parents=True)
        make_x3p(root, "measurement.x3p", nx=2, ny=2)

        cfg = ConversionConfig(
            root=tmp_path / "root",
            output_dir=tmp_path / "out",
            api_url="http://api:8000",
            force=True,
        )

        post_resp = MagicMock()
        post_resp.json.return_value = {
            "preview": "http://api:8000/files/preview.png",
            "surface_map": "http://api:8000/files/surface_map.png",
        }
        get_resp = MagicMock()
        get_resp.content = b"image_bytes"

        with (
            patch("scripts.convert_matlab_results.requests.post", return_value=post_resp),
            patch("scripts.convert_matlab_results.requests.get", return_value=get_resp) as mock_get,
        ):
            result = convert_measurement_x3p(root, cfg)

        assert mock_get.call_count == 2  # noqa: PLR2004
        out_dir = result.parent
        assert (out_dir / "preview.png").read_bytes() == b"image_bytes"
        assert (out_dir / "surface_map.png").read_bytes() == b"image_bytes"


class TestConvertMark:
    """Tests for :func:`convert_mark`."""

    def test_skips_existing(self, tmp_path):
        """Skip mark conversion when ``mark.json`` already exists in output."""
        root = tmp_path / "root" / "meas" / "mark"
        root.mkdir(parents=True)
        (root / "mark.mat").touch()

        out_mark = tmp_path / "out" / "meas" / "mark"
        out_mark.mkdir(parents=True)
        (out_mark / "mark.json").write_text("{}")

        cfg = ConversionConfig(root=tmp_path / "root", output_dir=tmp_path / "out", api_url="http://x")
        convert_mark(root, Path("dummy.x3p"), cfg)

    def test_impression_mark_full_flow(self, tmp_path):
        """Process an impression mark: extract params, call API, download files."""
        root = tmp_path / "root"
        mark_folder = root / "meas" / "mark1"
        mark_folder.mkdir(parents=True)
        (mark_folder / "mark.mat").touch()
        make_x3p(mark_folder.parent, "measurement.x3p", nx=200, ny=160)

        cfg = ConversionConfig(root=root, output_dir=tmp_path / "out", api_url="http://api:8000")
        converted_x3p = tmp_path / "converted.x3p"

        mock_mark_type = MagicMock()
        mock_mark_type.is_impression.return_value = True
        mock_mark_type.value = "breech_face"

        post_resp = MagicMock()
        post_resp.json.return_value = {
            "mark_json": "http://api:8000/files/mark.json",
            "mark_npz": "http://api:8000/files/mark.npz",
            "processed": "http://api:8000/files/leveled.png",
            "status_code": 200,
        }
        get_resp = MagicMock()
        get_resp.content = b"file_data"

        with (
            patch("scripts.convert_matlab_results.load_mat_struct", return_value=MagicMock()),
            patch("scripts.convert_matlab_results.extract_mark_type", return_value=mock_mark_type),
            patch("scripts.convert_matlab_results.get_x3p_shape", return_value=(200, 160)),
            patch(
                "scripts.convert_matlab_results.extract_mask_and_bounding_box",
                return_value=(np.ones((160, 200), dtype=bool), None),
            ),
            patch("scripts.convert_matlab_results.extract_impression_params", return_value={"pixel_size": 1.0}),
            patch("scripts.convert_matlab_results.requests.post", return_value=post_resp) as mock_post,
            patch("scripts.convert_matlab_results.requests.get", return_value=get_resp),
        ):
            convert_mark(mark_folder, converted_x3p, cfg)

        assert "impression" in mock_post.call_args.args[0]
        mark_dir = cfg.output_dir / "meas" / "mark1"
        assert (mark_dir / "mark.json").read_bytes() == b"file_data"
        assert (mark_dir / "mark.npz").read_bytes() == b"file_data"
        assert (mark_dir / "processed" / "leveled.png").read_bytes() == b"file_data"

    def test_striation_mark_with_bounding_box(self, tmp_path):
        """Process a striation mark and include bounding_box_list in the request."""
        root = tmp_path / "root"
        mark_folder = root / "meas" / "mark1"
        mark_folder.mkdir(parents=True)
        (mark_folder / "mark.mat").touch()
        make_x3p(mark_folder.parent, "measurement.x3p", nx=100, ny=100)

        cfg = ConversionConfig(root=root, output_dir=tmp_path / "out", api_url="http://api:8000")

        mock_mark_type = MagicMock()
        mock_mark_type.is_impression.return_value = False
        mock_mark_type.value = "striation"
        bbox = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)

        post_resp = MagicMock()
        post_resp.json.return_value = {}

        with (
            patch("scripts.convert_matlab_results.load_mat_struct", return_value=MagicMock()),
            patch("scripts.convert_matlab_results.extract_mark_type", return_value=mock_mark_type),
            patch("scripts.convert_matlab_results.get_x3p_shape", return_value=(100, 100)),
            patch(
                "scripts.convert_matlab_results.extract_mask_and_bounding_box",
                return_value=(np.ones((100, 100), dtype=bool), bbox),
            ),
            patch("scripts.convert_matlab_results.extract_striation_params", return_value={"highpass_cutoff": 2e-3}),
            patch("scripts.convert_matlab_results.requests.post", return_value=post_resp) as mock_post,
        ):
            convert_mark(mark_folder, tmp_path / "x.x3p", cfg)

        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "bounding_box_list" in body
        assert "striation" in mock_post.call_args.args[0]


class TestRunParallel:
    """Tests for :func:`_run_parallel`."""

    def test_runs_tasks_and_collects_results(self):
        """Execute tasks in parallel and return a dict keyed by task key."""
        tasks = [
            ("a", lambda x, y: x + y, (1, 2)),
            ("b", lambda x, y: x * y, (3, 4)),
        ]
        results = _run_parallel(tasks, workers=2, desc="test", unit=" items")
        assert results == {"a": 3, "b": 12}

    def test_empty_tasks(self):
        """Return an empty dict when no tasks are given."""
        assert _run_parallel([], workers=1, desc="empty", unit=" x") == {}


class TestMain:
    """Tests for :func:`main`."""

    @patch("scripts.convert_matlab_results._run_parallel")
    @patch("scripts.convert_matlab_results.copy_db_scratch_files", return_value=0)
    @patch("scripts.convert_matlab_results.find_mark_folders", return_value=iter([]))
    def test_no_marks(self, mock_find, mock_copy, mock_parallel, tmp_path):
        """Run the pipeline with no marks found."""
        root = tmp_path / "root"
        root.mkdir()
        output = tmp_path / "output"

        with patch("sys.argv", ["prog", str(root), str(output)]):
            main()

        assert output.exists()
        mock_copy.assert_called_once()

    @patch("scripts.convert_matlab_results._run_parallel")
    @patch("scripts.convert_matlab_results.copy_db_scratch_files", return_value=0)
    @patch("scripts.convert_matlab_results.find_mark_folders")
    def test_with_marks(self, mock_find, mock_copy, mock_parallel, tmp_path):
        """Run the pipeline with marks, calling _run_parallel twice."""
        root = tmp_path / "root"
        root.mkdir()
        output = tmp_path / "output"

        meas = root / "case1"
        mark = meas / "mark1"
        mock_find.return_value = iter([(meas, mark)])
        mock_parallel.side_effect = [{meas: Path("/out/measurement.x3p")}, {}]

        with patch("sys.argv", ["prog", str(root), str(output), "--force", "--workers", "2"]):
            main()

        assert mock_parallel.call_count == 2  # noqa: PLR2004
