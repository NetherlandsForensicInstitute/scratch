import pytest
from pydantic import ValidationError
from container_models.base import ConfigBaseModel
from container_models.scan_image import ScanImage


class TestModel(ConfigBaseModel):
    string_field: str
    float_field: float


@pytest.fixture(scope="module")
def test_model() -> TestModel:
    return TestModel(string_field="some_string", float_field=100.0)


def test_model_copy_updates_correct_fields(test_model: TestModel):
    updated = test_model.model_copy(update={"float_field": 150.0})
    assert updated.float_field == 150.0


def test_model_copy_raises_on_incorrect_fields(test_model: TestModel):
    with pytest.raises(ValidationError):
        test_model.model_copy(update={"float_field": "invalid_string"})


def test_scan_image_model_copy_validates_updated_fields(
    scan_image_with_nans: ScanImage,
):
    with pytest.raises(ValidationError):
        scan_image_with_nans.model_copy(
            update={"data": scan_image_with_nans.data.flatten()}
        )
    with pytest.raises(TypeError):
        scan_image_with_nans.model_copy(update={"data": [[1, 2], [3, 4]]})
    with pytest.raises(TypeError):
        scan_image_with_nans.model_copy(update={"data": "1"})
