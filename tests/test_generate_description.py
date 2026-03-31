from pydantic import BaseModel, Field

from schemas import generate_description


class UserModel(BaseModel):
    name: str = Field(..., description="User full name")
    age: int = Field(18, description="User age")
    nickname: str | None = Field(None, description="Optional nickname")


def test_generate_description_distinguishes_required_and_optional_fields():
    # Arrange
    expected_field_requirements = [
        "`name` (required)",
        "`age` (optional)",
        "`nickname` (optional)",
    ]
    # Act
    result = generate_description(UserModel)
    # Assert
    for field_requirement in expected_field_requirements:
        assert field_requirement in result, "Optional fields should be marked as optional"


def test_generate_description_has_formatted_field():
    # Arrange
    expected_field = "- `name` (required)\n  User full name"
    # Act
    result = generate_description(UserModel)
    # Assert
    assert expected_field in result, "Field should be formatted as required"


def test_generate_description_has_beginning():
    # Arrange
    class SmallModel(BaseModel):
        name: str = Field(..., description="User full name")

    # Act
    result = generate_description(SmallModel)
    # Assert
    assert "\n\n---\n\nExpected form data:\n- `name` (required)\n  User full name" == result, (
        "It should start with expected header"
    )
