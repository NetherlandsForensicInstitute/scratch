import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK

from dependencies import get_tmp_dir
from main import app


def test_get_tmp_dir_returns_correct_directory() -> None:
    """Test that get_tmp_dir returns the a temporary directory."""
    # Arrange
    app = FastAPI()
    app.state.temp_dir = tempfile.TemporaryDirectory()
    expected = Path(app.state.temp_dir.name)
    request = Request({"type": "http", "app": app})
    # Act
    result = get_tmp_dir(request)
    # Assert
    assert result == expected


def test_get_tmp_dir_returns_the_same_directory_multiple_times() -> None:
    """Test that get_tmp_dir doesn't change temp directory."""
    # Arrange
    app = FastAPI()
    app.state.temp_dir = tempfile.TemporaryDirectory()
    expected = Path(app.state.temp_dir.name)
    request = Request({"type": "http", "app": app})
    # Act
    responses = [get_tmp_dir(request) for _ in range(5)]
    # Assert
    assert all(response == expected for response in responses)


def create_test_app(tmp_dir: Path) -> FastAPI:
    _app = FastAPI()
    _app.state.temp_dir = tmp_dir

    @_app.get("/tmp-dir")  # noqa: S108
    def tmp_dir_endpoint(tmp_dir: Path = Depends(get_tmp_dir)):
        return {"tmp_dir": str(tmp_dir)}

    return _app


def test_get_tmp_dir_with_testclient(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.get("/tmp-dir")  # noqa: S108

    assert response.status_code == HTTP_200_OK

    returned = Path(response.json()["tmp_dir"])
    expected = Path(app.state.temp_dir.name)

    assert returned == expected


@app.get("/tmp1")  # noqa: S108
def tmp1(tmp: Path = Depends(get_tmp_dir)):
    return {"tmp": str(tmp)}


@app.get("/tmp2")  # noqa: S108
def tmp2(tmp: Path = Depends(get_tmp_dir)):
    return {"tmp": str(tmp)}


def test_tmp_dir_same_path_across_calls():
    with TestClient(app) as client:
        results = [client.get("/tmp1").json()["tmp"] for _ in range(3)]  # noqa: S108

        assert all(result == results[0] for result in results), "Temp dir should remain constant during lifespan"
        assert all(Path(result).exists() for result in results)


def test_tmp_dir_deleted_after_shutdown():
    with TestClient(app) as client:
        tmp = Path(client.get("/tmp1").json()["tmp"])  # noqa: S108
        assert tmp.exists()
    assert not tmp.exists(), "Temp dir should be removed after app shutdown"


def test_tmp_dir_is_different_after_restart():
    with TestClient(app) as client1:
        path1 = client1.get("/tmp1").json()["tmp"]  # noqa: S108
    with TestClient(app) as client2:
        path2 = client2.get("/tmp1").json()["tmp"]  # noqa: S108
    assert path1 != path2, "Temp dir should be regenerated per lifespan run"


def test_multiple_endpoints_share_same_temp_dir():
    with TestClient(app) as client:
        p1 = client.get("/tmp1").json()["tmp"]  # noqa: S108
        p2 = client.get("/tmp2").json()["tmp"]  # noqa: S108

        assert p1 == p2, "All endpoints should see the same app.state.temp_dir"
        assert Path(p1).exists()
