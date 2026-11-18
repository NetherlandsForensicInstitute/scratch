import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from lir.data.models import FeatureData
from lir.lrsystems import LRSystem
from lrmodule import get_lr_system

from conversion.likelihood_ratio import (
    DummyLRSystem,
    ModelSpecs,
    get_reference_data_from_path,
)
from conversion.likelihood_ratio.calculations import (
    calculate_lr_impression,
    calculate_lr_striation,
)

# Allow tiny numerical noise when checking that LLRs never decrease.
MONOTONIC_TOL = 1e-9
MODELS_DIR = Path(
    os.environ.get("SCRATCH_LR_MODELS_DIR", "/mnt/scratch/dev_lr_modellen/2026-06-23")
)


@pytest.fixture(scope="session")
def impression_lr_system_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    lr_dir = tmp_path_factory.mktemp("lr_impression")

    # Dummy model
    with (lr_dir / "model.pkl").open("wb") as f:
        pickle.dump(DummyLRSystem(), f)

    # Dummy reference data
    (lr_dir / "reference_data.csv").write_text(
        "feature1,feature2,hypothesis\n0.8,0.9,1\n0.85,0.95,1\n0.7,0.88,1\n0.1,0.2,0\n0.15,0.25,0\n0.05,0.1,0\n"
    )

    return lr_dir


@pytest.fixture(scope="session")
def striation_lr_system_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    lr_dir = tmp_path_factory.mktemp("lr_striation")

    with (lr_dir / "model.pkl").open("wb") as f:
        pickle.dump(DummyLRSystem(), f)

    (lr_dir / "reference_data.csv").write_text(
        "feature1,hypothesis\n0.9,1\n0.85,1\n0.78,1\n0.3,0\n0.25,0\n0.15,0\n"
    )

    return lr_dir


@pytest.fixture
def dummy_lr_system() -> DummyLRSystem:
    return DummyLRSystem()


def _load_model(name: str) -> LRSystem:
    folder = MODELS_DIR / name
    if not (folder / "model.pkl").exists():
        pytest.skip(f"no trained model at {folder} (set SCRATCH_LR_MODELS_DIR)")
    return get_lr_system(folder)


@pytest.fixture(scope="module")
def striation_model() -> LRSystem:
    return _load_model("aperture_shear")


@pytest.fixture(scope="module")
def impression_model() -> LRSystem:
    return _load_model("firing_pin_impression")


@pytest.fixture
def striation_reference_data() -> ModelSpecs:
    return ModelSpecs(
        km_scores=np.array([0.9, 0.85, 0.78]),
        km_llrs=np.array([2.1, 1.8, 1.5]),
        km_llr_intervals=np.array([[1.9, 2.3], [1.6, 2.0], [1.3, 1.7]]),
        knm_scores=np.array([0.3, 0.25, 0.15, 0.1]),
        knm_llrs=np.array([-1.2, -0.9, -1.5, -2.0]),
        knm_llr_intervals=np.array(
            [[-1.4, -1.0], [-1.1, -0.7], [-1.7, -1.3], [-2.2, -1.8]]
        ),
    )


class TestGetReferenceDataFromPath:
    @pytest.mark.parametrize(
        "lr_path_fixture",
        ["striation_lr_system_path", "impression_lr_system_path"],
    )
    def test_splits_km_and_knm_by_label(self, lr_path_fixture, request):
        lr_path = request.getfixturevalue(lr_path_fixture)
        result = get_reference_data_from_path(lr_path)

        assert result.km_scores.shape[0] == 3
        assert result.knm_scores.shape[0] == 3
        assert result.km_llrs.shape[0] == 3
        assert result.knm_llrs.shape[0] == 3

    def test_raises_when_labels_are_none(self, tmp_path):
        mock_ref = MagicMock(spec=FeatureData)
        mock_ref.labels = None

        with (
            patch(
                "conversion.likelihood_ratio.data_loaders.get_lr_system",
                return_value=DummyLRSystem(),
            ),
            patch(
                "conversion.likelihood_ratio.data_loaders.get_reference_data",
                return_value=mock_ref,
            ),
            pytest.raises(ValueError, match="reference data must have labels"),
        ):
            get_reference_data_from_path(tmp_path)


class TestModelSpecs:
    @pytest.mark.parametrize(
        "field, match",
        [
            ("km", "km_scores and km_lrs must have the same length"),
            ("knm", "knm_scores and knm_lrs must have the same length"),
        ],
    )
    def test_length_mismatch_raises(self, field, match):
        kwargs = dict(
            km_scores=np.array([0.9]),
            km_llrs=np.array([1.0]),
            km_llr_intervals=None,
            knm_scores=np.array([0.1]),
            knm_llrs=np.array([-1.0]),
            knm_llr_intervals=None,
        )
        kwargs[f"{field}_scores"] = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match=match):
            ModelSpecs(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["km_llr_intervals", "knm_llr_intervals"])
    def test_llr_intervals_raises_when_none(self, striation_reference_data, field):
        data = {
            "km_scores": striation_reference_data.km_scores,
            "km_llrs": striation_reference_data.km_llrs,
            "km_llr_intervals": striation_reference_data.km_llr_intervals,
            "knm_scores": striation_reference_data.knm_scores,
            "knm_llrs": striation_reference_data.knm_llrs,
            "knm_llr_intervals": striation_reference_data.knm_llr_intervals,
            field: None,
        }
        specs = ModelSpecs(**data)  # type: ignore[arg-type]
        with pytest.raises(
            ValueError, match="Only models with llr_intervals can be used"
        ):
            specs.llr_intervals


class TestCalculateLrStriation:
    @pytest.mark.parametrize("score", [-0.99, -0.5, 0.0, 0.5, 0.9, 0.999])
    def test_passes_score_as_single_feature(self, dummy_lr_system, score):
        calculate_lr_striation(dummy_lr_system, score)

        (features,) = dummy_lr_system.received
        assert features.shape == (1, 1)
        assert features[0, 0] == pytest.approx(score)

    def test_returns_lr_system_output(self, dummy_lr_system):
        result = calculate_lr_striation(dummy_lr_system, 0.5)

        np.testing.assert_array_equal(result.llrs, [0.0])
        np.testing.assert_array_equal(result.llr_intervals, [[-1.0, 1.0]])

    @pytest.mark.parametrize("score", [-1.0, 1.0])
    def test_passes_endpoints_unchanged(self, dummy_lr_system, score):
        calculate_lr_striation(dummy_lr_system, score)

        assert dummy_lr_system.received[0][0, 0] == score

    @pytest.mark.parametrize("score", [1.0001, -1.0001, 2.0, 85.0, -100.0, np.nan])
    def test_rejects_scores_outside_range(self, dummy_lr_system, score):
        with pytest.raises(ValueError, match="between -1 and 1"):
            calculate_lr_striation(dummy_lr_system, score)
        assert dummy_lr_system.received == []


class TestCalculateLrImpression:
    @pytest.mark.parametrize(("score", "n_cells"), [(0, 20), (5, 20), (20, 20), (1, 1)])
    def test_passes_k_then_n(self, dummy_lr_system, score, n_cells):
        calculate_lr_impression(dummy_lr_system, score, n_cells)

        (features,) = dummy_lr_system.received
        np.testing.assert_array_equal(features, [[score, n_cells]])

    def test_returns_lr_system_output(self, dummy_lr_system):
        result = calculate_lr_impression(dummy_lr_system, 5, 20)

        np.testing.assert_array_equal(result.llrs, [0.0])
        np.testing.assert_array_equal(result.llr_intervals, [[-1.0, 1.0]])

    @pytest.mark.parametrize(("score", "n_cells"), [(-1, 20), (21, 20), (5, 4)])
    def test_rejects_score_outside_zero_to_n(self, dummy_lr_system, score, n_cells):
        with pytest.raises(ValueError, match="CMC count"):
            calculate_lr_impression(dummy_lr_system, score, n_cells)
        assert dummy_lr_system.received == []

    @pytest.mark.parametrize("n_cells", [0, -5])
    def test_rejects_non_positive_n_cells(self, dummy_lr_system, n_cells):
        with pytest.raises(ValueError, match="n_cells"):
            calculate_lr_impression(dummy_lr_system, 0, n_cells)
        assert dummy_lr_system.received == []


@pytest.mark.integration
class TestStriationModel:
    @pytest.mark.parametrize("score", [0.0, 0.5, 0.9, 0.999])
    def test_gives_one_finite_llr(self, striation_model, score):
        result = calculate_lr_striation(striation_model, score)

        assert result.llrs.shape == (1,)
        assert np.isfinite(result.llrs).all()

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="lr_module transform_marktype_ccf does not handle ccf = ±1 (log10 of 0); should clip",
    )
    @pytest.mark.parametrize(("score", "nearby"), [(1.0, 0.999), (-1.0, -0.999)])
    def test_endpoints_match_nearby_scores(self, striation_model, score, nearby):
        at_endpoint = calculate_lr_striation(striation_model, score).llrs[0]
        near_endpoint = calculate_lr_striation(striation_model, nearby).llrs[0]

        assert at_endpoint == pytest.approx(near_endpoint)

    @pytest.mark.xfail(
        strict=True,
        reason="KDE tail artifact: LLR decreases between ccf 0.0 and ~0.06 (model issue)",
    )
    def test_monotonic_for_positive_scores(self, striation_model):
        scores = np.linspace(0.0, 0.999, 50)
        llrs = np.array(
            [calculate_lr_striation(striation_model, s).llrs[0] for s in scores]
        )

        assert np.all(np.diff(llrs) >= -MONOTONIC_TOL), (
            "LLR decreases while score increases"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="KDE tail artifact: negative ccf gives maximum same-source LLR (model issue)",
    )
    def test_monotonic_over_full_range(self, striation_model):
        scores = np.linspace(-0.99, 0.999, 100)
        llrs = np.array(
            [calculate_lr_striation(striation_model, s).llrs[0] for s in scores]
        )

        assert np.all(np.diff(llrs) >= -MONOTONIC_TOL), (
            "LLR decreases while score increases"
        )


@pytest.mark.integration
class TestImpressionModel:
    def test_monotonic_in_matching_cells(self, impression_model):
        n_cells = 20
        llrs = np.array(
            [
                calculate_lr_impression(impression_model, k, n_cells).llrs[0]
                for k in range(n_cells + 1)
            ]
        )

        assert np.isfinite(llrs).all()
        assert np.all(np.diff(llrs) >= -MONOTONIC_TOL), (
            "LLR decreases while the number of matching cells increases"
        )
        assert llrs[0] < 0 < llrs[-1]
