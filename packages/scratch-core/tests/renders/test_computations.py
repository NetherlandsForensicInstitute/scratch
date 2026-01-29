import pytest

from container_models.base import Point
from renders.computations import clip_factors


@pytest.mark.parametrize(
    "factors,preserve_aspect_ratio,expected",
    [
        pytest.param(
            Point[float](2.0, 1.5),
            False,
            Point[float](2.0, 1.5),
            id="no_clipping_needed",
        ),
        pytest.param(
            Point[float](0.5, 2.0),
            False,
            Point[float](1.0, 2.0),
            id="clip_below_one",
        ),
        pytest.param(
            Point[float](2.0, 1.5),
            True,
            Point[float](2.0, 2.0),
            id="preserve_aspect_ratio_clips_to_max",
        ),
        pytest.param(
            Point[float](0.5, 2.0),
            True,
            Point[float](2.0, 2.0),
            id="preserve_aspect_ratio_clip_below_one_to_max",
        ),
        pytest.param(
            Point[float](0.5, 0.8),
            True,
            Point[float](1.0, 1.0),
            id="preserve_aspect_ratio_all_below_one",
        ),
    ],
)
def test_clip_factors(
    factors: Point[float],
    preserve_aspect_ratio: bool,
    expected: Point[float],
) -> None:
    assert clip_factors(factors, preserve_aspect_ratio) == expected


def test_clip_factor_logs_when_preserve_aspact_ratio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    min_factor = 0.5
    max_factor = 2.0

    # Act
    _ = clip_factors(factors=Point(min_factor, max_factor), preserve_aspect_ratio=True)

    # Assert
    assert (
        f"Preserving aspact ratio in clip factors, max point: {max_factor}"
        in caplog.messages
    )
