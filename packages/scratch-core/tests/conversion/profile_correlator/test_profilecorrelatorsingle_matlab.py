"""
Tests comparing Python ProfileCorrelatorSingle output with MATLAB reference.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from conversion.profile_correlator import (
    AlignmentParameters,
    ComparisonResults,
    Profile,
    correlate_profiles,
)


@dataclass
class ProfileCorrelatorParams:
    """Parameters for ProfileCorrelatorSingle (test helper)."""

    part_mark_perc: float = 8.0
    pass_freqs: list[float] = field(
        default_factory=lambda: [1000, 500, 250, 100, 50, 25, 10, 5]
    )
    filtertype: str = "lowpass"
    remove_zeros: bool = True
    show_info: bool = False
    plot_figures: bool = False
    x0: tuple[float, float] = (0.0, 0.0)
    use_mean: bool = True
    cutoff_hi: float = 1000.0
    cutoff_lo: float = 5.0
    max_translation: float = 1e7
    max_scaling: float = 0.05
    inclusion_threshold: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileCorrelatorParams":
        """Create params from dictionary."""
        # Handle 'pass' key which is a Python reserved word
        pass_freqs = d.get("pass", [1000, 500, 250, 100, 50, 25, 10, 5])
        if isinstance(pass_freqs, list) and len(pass_freqs) > 0:
            # Flatten if nested
            if isinstance(pass_freqs[0], list):
                pass_freqs = [p[0] for p in pass_freqs]

        x0 = d.get("x0", [0, 0])
        if isinstance(x0, list):
            x0 = tuple(x0)

        return cls(
            part_mark_perc=d.get("part_mark_perc", 8.0),
            pass_freqs=pass_freqs,
            filtertype=d.get("filtertype", "lowpass"),
            remove_zeros=bool(d.get("remove_zeros", 1)),
            show_info=bool(d.get("show_info", 0)),
            plot_figures=bool(d.get("plot_figures", 0)),
            x0=x0,
            use_mean=bool(d.get("use_mean", 1)),
            cutoff_hi=d.get("cutoff_hi", 1000.0),
            cutoff_lo=d.get("cutoff_lo", 5.0),
            max_translation=d.get("max_translation", 1e7),
            max_scaling=d.get("max_scaling", 0.05),
            inclusion_threshold=d.get("inclusion_threshold", 0.5),
        )

    def to_alignment_params(self) -> AlignmentParameters:
        """Convert to AlignmentParameters for the Python implementation."""
        return AlignmentParameters(
            scale_passes=tuple(self.pass_freqs),
            max_translation=self.max_translation,
            max_scaling=self.max_scaling,
            cutoff_hi=self.cutoff_hi,
            cutoff_lo=self.cutoff_lo,
            partial_mark_threshold=self.part_mark_perc,
            inclusion_threshold=self.inclusion_threshold,
            use_mean=self.use_mean,
            remove_boundary_zeros=self.remove_zeros,
        )


@dataclass
class ResultsTable:
    """Results table from ProfileCorrelatorSingle (test helper)."""

    bProfile: int = 0
    bSegments: int = 0
    bPartialProfile: int = 0
    vPixSep1: float = 0.0
    vPixSep2: float = 0.0
    pOverlap: float = 0.0
    lOverlap: float = 0.0
    xcorr: float | None = None
    xcorr_max: float | None = None
    trans_shift: float | None = None
    trans_scale: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ResultsTable":
        """Create results table from dictionary."""
        if d.get("is_empty", False):
            return cls()

        return cls(
            bProfile=int(d.get("bProfile", 0)),
            bSegments=int(d.get("bSegments", 0)),
            bPartialProfile=int(d.get("bPartialProfile", 0)),
            vPixSep1=float(d.get("vPixSep1", 0.0)),
            vPixSep2=float(d.get("vPixSep2", 0.0)),
            pOverlap=float(d.get("pOverlap", 0.0)),
            lOverlap=float(d.get("lOverlap", 0.0)),
            xcorr=d.get("xcorr"),
            xcorr_max=d.get("xcorr_max"),
            trans_shift=d.get("trans_shift"),
            trans_scale=d.get("trans_scale"),
        )

    @classmethod
    def from_comparison_results(cls, results: ComparisonResults) -> "ResultsTable":
        """Convert ComparisonResults to ResultsTable for comparison."""
        return cls(
            bProfile=1 if results.is_profile_comparison else 0,
            bSegments=0,  # Not used in Python implementation
            bPartialProfile=1 if results.is_partial_profile else 0,
            vPixSep1=results.pixel_size_ref,
            vPixSep2=results.pixel_size_comp,
            pOverlap=results.overlap_ratio,
            lOverlap=results.overlap_length,
            xcorr=results.correlation_coefficient,
            xcorr_max=results.correlation_coefficient,  # Same in current implementation
            trans_shift=results.position_shift,
            trans_scale=results.scale_factor,
        )


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB test case."""

    name: str
    # Input profiles
    profile_ref_data: np.ndarray
    profile_comp_data: np.ndarray
    ref_pixel_size: float  # In meters
    comp_pixel_size: float  # In meters
    # Parameters
    params: ProfileCorrelatorParams
    iVerbose: int
    # Input results table (may be empty)
    input_results_table: ResultsTable | None
    # Expected output
    expected_results: ResultsTable
    # Metadata
    length_diff_percentage: float
    is_partial_expected: bool
    ref_has_nans: bool
    comp_has_nans: bool

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        # Load profile arrays
        profile_ref_data = np.load(
            case_dir / "input_profile_ref.npy", allow_pickle=True
        )
        profile_comp_data = np.load(
            case_dir / "input_profile_comp.npy", allow_pickle=True
        )

        # Parse parameters
        params = ProfileCorrelatorParams.from_dict(meta.get("params", {}))

        # Parse input results table
        input_rt_dict = meta.get("input_results_table", {"is_empty": True})
        input_results_table = None
        if not input_rt_dict.get("is_empty", True):
            input_results_table = ResultsTable.from_dict(input_rt_dict)

        # Parse expected output results table
        expected_results = ResultsTable.from_dict(meta["output_results_table"])

        return cls(
            name=case_dir.name,
            profile_ref_data=profile_ref_data,
            profile_comp_data=profile_comp_data,
            ref_pixel_size=meta["ref_xdim"],  # xdim is pixel size in meters
            comp_pixel_size=meta["comp_xdim"],
            params=params,
            iVerbose=meta.get("iVerbose", 0),
            input_results_table=input_results_table,
            expected_results=expected_results,
            length_diff_percentage=meta.get("length_diff_percentage", 0.0),
            is_partial_expected=meta.get("is_partial_expected", False),
            ref_has_nans=meta.get("ref_has_nans", False),
            comp_has_nans=meta.get("comp_has_nans", False),
        )

    @property
    def has_nans(self) -> bool:
        """Check if either profile has NaN values."""
        return self.ref_has_nans or self.comp_has_nans

    @property
    def is_partial(self) -> bool:
        """Check if this is a partial profile comparison."""
        return self.is_partial_expected

    def get_profile_ref(self) -> Profile:
        """Create Profile object for reference."""
        return Profile(
            depth_data=self.profile_ref_data.astype(np.float64),
            pixel_size=self.ref_pixel_size,
            cutoff_hi=self.params.cutoff_hi,
            cutoff_lo=self.params.cutoff_lo,
        )

    def get_profile_comp(self) -> Profile:
        """Create Profile object for comparison."""
        return Profile(
            depth_data=self.profile_comp_data.astype(np.float64),
            pixel_size=self.comp_pixel_size,
            cutoff_hi=self.params.cutoff_hi,
            cutoff_lo=self.params.cutoff_lo,
        )


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in a directory."""
    if not test_cases_dir.exists():
        return []
    return [
        MatlabTestCase.from_directory(d)
        for d in sorted(test_cases_dir.iterdir())
        if d.is_dir() and (d / "metadata.json").exists()
    ]


def run_python_profile_correlator(test_case: MatlabTestCase) -> ResultsTable:
    """
    Run Python ProfileCorrelatorSingle and return results.
    """
    # Create Profile objects
    profile_ref = test_case.get_profile_ref()
    profile_comp = test_case.get_profile_comp()

    # Convert test params to AlignmentParameters
    params = test_case.params.to_alignment_params()

    # Run the correlator
    results = correlate_profiles(profile_ref, profile_comp, params)

    # Convert to ResultsTable for comparison
    return ResultsTable.from_comparison_results(results)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_cases_dir() -> Path:
    """Path to test cases directory."""
    return Path(__file__).parent.parent / "resources" / "profile_correlator"


@pytest.fixture(scope="module")
def all_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Load all MATLAB test cases."""
    cases = discover_test_cases(test_cases_dir)
    if not cases:
        pytest.skip(f"No test cases found in {test_cases_dir}")
    return cases


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically."""
    if "test_case_name" not in metafunc.fixturenames:
        return

    test_cases_dir = Path(__file__).parent.parent / "resources" / "profile_correlator"
    cases = discover_test_cases(test_cases_dir)
    metafunc.parametrize("test_case_name", [c.name for c in cases])


@pytest.fixture
def test_case(
    test_case_name: str, all_test_cases: list[MatlabTestCase]
) -> MatlabTestCase:
    """Get individual test case by name."""
    for case in all_test_cases:
        if case.name == test_case_name:
            return case
    pytest.skip(f"Test case {test_case_name} not found")


# ============================================================================
# Helper Functions
# ============================================================================


def assert_close(
    actual: float | None,
    expected: float | None,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> None:
    """Assert two values are close, handling None values."""
    if expected is None and actual is None:
        return
    if expected is None or actual is None:
        pytest.fail(f"{name}: expected {expected}, got {actual}")

    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        rel_diff = abs(actual - expected) / max(abs(expected), 1e-10)
        pytest.fail(
            f"{name}: expected {expected:.6f}, got {actual:.6f} "
            f"(rel_diff={rel_diff:.2e})"
        )


def assert_equal_int(actual: int, expected: int, name: str) -> None:
    """Assert two integer values are equal."""
    if actual != expected:
        pytest.fail(f"{name}: expected {expected}, got {actual}")


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> None:
    """Assert two arrays are close, handling NaN values."""
    if actual.shape != expected.shape:
        pytest.fail(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")

    # Check NaN positions match
    actual_nans = np.isnan(actual)
    expected_nans = np.isnan(expected)
    if not np.array_equal(actual_nans, expected_nans):
        pytest.fail(f"{name}: NaN positions don't match")

    # Compare non-NaN values
    mask = ~actual_nans
    if not np.allclose(actual[mask], expected[mask], rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(actual[mask] - expected[mask]))
        pytest.fail(f"{name}: arrays not close, max_diff={max_diff:.2e}")


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.matlab
class TestProfileCorrelatorSingleMatlabComparison:
    """Test Python ProfileCorrelatorSingle against MATLAB reference outputs."""

    # Tolerance thresholds
    CORRELATION_RTOL = 1e-3  # Relative tolerance for correlation values
    OVERLAP_RTOL = 1e-4  # Relative tolerance for overlap values
    PIXSEP_RTOL = 1e-6  # Relative tolerance for pixel separation

    def test_matlab_comparison(self, test_case: MatlabTestCase):
        """Test that Python output matches MATLAB reference."""
        # Run Python implementation
        python_results = run_python_profile_correlator(test_case)
        expected = test_case.expected_results

        # Check integer flags
        assert_equal_int(
            python_results.bProfile,
            expected.bProfile,
            f"{test_case.name}: bProfile",
        )
        assert_equal_int(
            python_results.bSegments,
            expected.bSegments,
            f"{test_case.name}: bSegments",
        )
        assert_equal_int(
            python_results.bPartialProfile,
            expected.bPartialProfile,
            f"{test_case.name}: bPartialProfile",
        )

        # Check pixel separations
        assert_close(
            python_results.vPixSep1,
            expected.vPixSep1,
            f"{test_case.name}: vPixSep1",
            rtol=self.PIXSEP_RTOL,
        )
        assert_close(
            python_results.vPixSep2,
            expected.vPixSep2,
            f"{test_case.name}: vPixSep2",
            rtol=self.PIXSEP_RTOL,
        )

        # Check overlap values
        assert_close(
            python_results.pOverlap,
            expected.pOverlap,
            f"{test_case.name}: pOverlap",
            rtol=self.OVERLAP_RTOL,
        )
        assert_close(
            python_results.lOverlap,
            expected.lOverlap,
            f"{test_case.name}: lOverlap",
            rtol=self.OVERLAP_RTOL,
        )

        # Check correlation values (if present)
        if expected.xcorr is not None:
            assert_close(
                python_results.xcorr,
                expected.xcorr,
                f"{test_case.name}: xcorr",
                rtol=self.CORRELATION_RTOL,
            )

        if expected.xcorr_max is not None:
            assert_close(
                python_results.xcorr_max,
                expected.xcorr_max,
                f"{test_case.name}: xcorr_max",
                rtol=self.CORRELATION_RTOL,
            )

    def test_partial_profile_flag(self, test_case: MatlabTestCase):
        """Test that partial profile flag is set correctly."""
        python_results = run_python_profile_correlator(test_case)

        expected_partial = 1 if test_case.is_partial_expected else 0
        assert_equal_int(
            python_results.bPartialProfile,
            expected_partial,
            f"{test_case.name}: bPartialProfile based on length_diff={test_case.length_diff_percentage:.2f}%",
        )


@pytest.mark.matlab
class TestProfileCorrelatorSingleEdgeCases:
    """Test edge cases for ProfileCorrelatorSingle."""

    def test_profiles_with_nans(self, test_case: MatlabTestCase):
        """Test that profiles with NaN values are handled correctly."""
        if not test_case.has_nans:
            pytest.skip("Test case has no NaN values")

        python_results = run_python_profile_correlator(test_case)
        expected = test_case.expected_results

        # Basic sanity checks
        assert python_results.bProfile == expected.bProfile
        assert not np.isnan(python_results.pOverlap), "pOverlap should not be NaN"

    def test_partial_profiles(self, test_case: MatlabTestCase):
        """Test partial profile alignment."""
        if not test_case.is_partial:
            pytest.skip("Test case is not a partial profile comparison")

        python_results = run_python_profile_correlator(test_case)
        expected = test_case.expected_results

        assert python_results.bPartialProfile == 1
        assert_close(
            python_results.pOverlap,
            expected.pOverlap,
            f"{test_case.name}: pOverlap (partial)",
            rtol=1e-3,
        )


# ============================================================================
# Standalone Test Runner
# ============================================================================


def run_all_tests_standalone(test_cases_dir: Path) -> None:
    """Run all tests without pytest (for debugging)."""
    cases = discover_test_cases(test_cases_dir)
    if not cases:
        print(f"No test cases found in {test_cases_dir}")
        return

    print(f"Found {len(cases)} test cases")
    print()

    passed = 0
    failed = 0
    skipped = 0

    for case in cases:
        print(f"Running: {case.name}")
        try:
            python_results = run_python_profile_correlator(case)
            expected = case.expected_results

            # Check key values
            errors = []

            if python_results.bPartialProfile != expected.bPartialProfile:
                errors.append(
                    f"bPartialProfile: {python_results.bPartialProfile} != {expected.bPartialProfile}"
                )

            if not np.isclose(python_results.pOverlap, expected.pOverlap, rtol=1e-4):
                errors.append(
                    f"pOverlap: {python_results.pOverlap:.6f} != {expected.pOverlap:.6f}"
                )

            if errors:
                print(f"  FAILED: {', '.join(errors)}")
                failed += 1
            else:
                print("  PASSED")
                passed += 1

        except NotImplementedError:
            print("  SKIPPED (not implemented)")
            skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])
    else:
        test_dir = Path(__file__).parent.parent / "resources" / "profile_correlator"

    run_all_tests_standalone(test_dir)
