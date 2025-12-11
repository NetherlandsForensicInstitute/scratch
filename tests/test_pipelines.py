from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi import HTTPException
from returns.io import impure_safe
from returns.result import safe

from pipelines import run_pipeline


@pytest.mark.parametrize(
    "pipeline",
    [
        pytest.param((safe(lambda x: x / 0),), id="force a runtime error"),
        pytest.param((impure_safe(lambda x: Path(str(x)).read_bytes()),), id="force on io error"),
    ],
)
def test_pipeline_failure_raises_http_exception(pipeline: tuple[Callable, ...]) -> None:
    """Test that pipeline failures raise HTTPException with status 500."""
    # Act & Assert
    with pytest.raises(HTTPException, match="Failed pipeline") as exc_info:
        run_pipeline(5, *pipeline, error_message="Failed pipeline")

    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
