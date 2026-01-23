"""
Tests comparing Python ProfileCorrelatorSingle output with MATLAB reference.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

# TODO: Update these imports to match your actual module structure
# from your_module.profile_correlator import profile_correlator_single
# from your_module.profile import Profile


@dataclass
class ProfileCorrelatorParams:
    """Parameters for ProfileCorrelatorSingle."""

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


@dataclass
class ResultsTable:
    """Results table from ProfileCorrelatorSingle."""

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


@dataclass
class Profile:
    """Profile data structure."""

    depth_data: np.ndarray
    xdim: float
    ydim: float

    @property
    def length(self) -> int:
        return len(self.depth_data)


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB test case."""

    name: str
    # Input profiles
    profile_ref: Profile
    profile_comp: Profile
    # Parameters
    params: ProfileCorrelatorParams
    iVerbose: int
    # Input results table (may be empty)
    input_results_table: ResultsTable | None
    # Expected output
    expected_results: ResultsTable
    # Output arrays (e.g., xcorr array at different scales)
    output_arrays: dict[str, np.ndarray]
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

        # Create Profile objects
        profile_ref = Profile(
            depth_data=profile_ref_data,
            xdim=meta["ref_xdim"],
            ydim=meta["ref_ydim"],
        )
        profile_comp = Profile(
            depth_data=profile_comp_data,
            xdim=meta["comp_xdim"],
            ydim=meta["comp_ydim"],
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

        # Load output arrays
        output_arrays = {}
        for array_name in meta.get("output_results_arrays", []):
            array_path = case_dir / f"output_results_{array_name}.npy"
            if array_path.exists():
                output_arrays[array_name] = np.load(array_path, allow_pickle=True)

        return cls(
            name=case_dir.name,
            profile_ref=profile_ref,
            profile_comp=profile_comp,
            params=params,
            iVerbose=meta.get("iVerbose", 0),
            input_results_table=input_results_table,
            expected_results=expected_results,
            output_arrays=output_arrays,
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


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in a directory."""
    if not test_cases_dir.exists():
        return []
    return [
        MatlabTestCase.from_directory(d)
        for d in sorted(test_cases_dir.iterdir())
        if d.is_dir() and (d / "metadata.json").exists()
    ]


def run_python_profile_correlator(
    test_case: MatlabTestCase,
) -> ResultsTable:
    """
    Run Python ProfileCorrelatorSingle and return results.

    TODO: Implement this function to call your Python implementation.
    """
    # Example implementation structure:
    #
    # from your_module import profile_correlator_single, ProfileCorrelatorResInit
    #
    # # Initialize results table if needed
    # if test_case.input_results_table is None:
    #     results_table = ProfileCorrelatorResInit()
    # else:
    #     results_table = convert_to_your_format(test_case.input_results_table)
    #
    # # Run the correlator
    # results = profile_correlator_single(
    #     profile_ref=test_case.profile_ref,
    #     profile_comp=test_case.profile_comp,
    #     results_table=results_table,
    #     param=test_case.params,
    #     iVerbose=test_case.iVerbose,
    # )
    #
    # return convert_results_to_dataclass(results)

    raise NotImplementedError(
        "Implement run_python_profile_correlator to call your Python implementation"
    )


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_cases_dir() -> Path:
    """Path to test cases directory."""
    # TODO: Update this path to match your project structure
    return Path(__file__).parent.parent / "resources" / "profile_correlator_single"


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

    # TODO: Update this path to match your project structure
    test_cases_dir = (
        Path(__file__).parent.parent / "resources" / "profile_correlator_single"
    )
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

    def test_output_arrays(self, test_case: MatlabTestCase):
        """Test that output arrays match MATLAB reference."""
        if not test_case.output_arrays:
            pytest.skip("Test case has no output arrays")

        # TODO: Implement this when your Python function returns arrays
        # python_results, python_arrays = run_python_profile_correlator_with_arrays(test_case)
        #
        # for array_name, expected_array in test_case.output_arrays.items():
        #     if array_name in python_arrays:
        #         assert_arrays_close(
        #             python_arrays[array_name],
        #             expected_array,
        #             f"{test_case.name}: {array_name}",
        #         )
        pytest.skip("Output array comparison not yet implemented")


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
        test_dir = Path("test_cases_ProfileCorrelatorSingle")

    run_all_tests_standalone(test_dir)
