import pickle
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from constants import ProcessorEndpoint, RoutePrefix
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    ImpressionLRParamaters,
    StriationLRParamaters,
)

ACCESS_URL = "http://localhost:8000"
EXPECTED_LR = 1.23


@pytest.fixture
def lr_system_pickle(tmp_path: Path) -> Path:
    """Trained LinearRegression pickle file."""
    model = LinearRegression()
    model.fit([[0.0], [1.0]], [0.0, 1.0])
    path = tmp_path / "lr_system.pkl"
    path.write_bytes(pickle.dumps(model))
    return path


@pytest.fixture
def mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Pair of temporary mark directories."""
    ref = tmp_path / "ref"
    ref.mkdir()
    comp = tmp_path / "comp"
    comp.mkdir()
    return ref, comp


@pytest.fixture(autouse=True)
def mock_directory_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace DirectoryAccess in the router with a mock that has a fixed access_url."""
    vault = MagicMock()
    vault.access_url = ACCESS_URL
    monkeypatch.setattr("processors.router.DirectoryAccess", lambda: vault)


@pytest.fixture(autouse=True)
def mock_calculate_lr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub calculate_lr to return a fixed value."""
    monkeypatch.setattr("processors.router.calculate_lr", lambda score, model: EXPECTED_LR)


@pytest.mark.integration
class TestCalculateLRImpressionEndpoint:
    """Tests for the calculate-lr-impression endpoint."""

    def test_returns_ok(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
        lr_system_pickle: Path,
    ) -> None:
        """Endpoint returns HTTP 200 for a valid impression LR request."""
        ref, comp = mark_dirs
        payload = CalculateLRImpression(
            mark_ref=ref,
            mark_comp=comp,
            score=5,
            lr_system=lr_system_pickle,
            n_cells=10,
            param=ImpressionLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/{RoutePrefix.PROCESSOR}/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=payload)

        assert response.status_code == HTTPStatus.OK, response.text

    def test_returns_lr_value(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
        lr_system_pickle: Path,
    ) -> None:
        """Response body contains the LR value returned by calculate_lr."""
        ref, comp = mark_dirs
        payload = CalculateLRImpression(
            mark_ref=ref,
            mark_comp=comp,
            score=5,
            lr_system=lr_system_pickle,
            n_cells=10,
            param=ImpressionLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/{RoutePrefix.PROCESSOR}/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}", json=payload)

        assert response.json()["lr"] == EXPECTED_LR


@pytest.mark.integration
class TestCalculateLRStriationEndpoint:
    """Tests for the calculate-lr-striation endpoint."""

    def test_returns_ok(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
        lr_system_pickle: Path,
    ) -> None:
        """Endpoint returns HTTP 200 for a valid striation LR request."""
        ref, comp = mark_dirs
        payload = CalculateLRStriation(
            mark_ref=ref,
            mark_comp=comp,
            score=5,
            lr_system=lr_system_pickle,
            param=StriationLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/{RoutePrefix.PROCESSOR}/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=payload)

        assert response.status_code == HTTPStatus.OK, response.text

    def test_returns_lr_value(
        self,
        client: TestClient,
        mark_dirs: tuple[Path, Path],
        lr_system_pickle: Path,
    ) -> None:
        """Response body contains the LR value returned by calculate_lr."""
        ref, comp = mark_dirs
        payload = CalculateLRStriation(
            mark_ref=ref,
            mark_comp=comp,
            score=5,
            lr_system=lr_system_pickle,
            param=StriationLRParamaters(),
        ).model_dump(mode="json")

        response = client.post(f"/{RoutePrefix.PROCESSOR}/{ProcessorEndpoint.CALCULATE_LR_STRIATION}", json=payload)

        assert response.json()["lr"] == EXPECTED_LR


def test_processors_placeholder(client: TestClient) -> None:
    """Test that the processor root endpoint redirects to documentation."""
    # Act
    response = client.get("/processor", follow_redirects=False)
    # Assert
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT, "endpoint should redirect"
    assert response.headers["location"] == "/docs#operations-tag-processor", "should redirect to processor docs"
