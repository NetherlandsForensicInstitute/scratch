from http import HTTPStatus
from unittest.mock import Mock

import pytest
from conversion.data_formats import Mark
from conversion.profile_correlator import Profile
from fastapi import HTTPException

from processors.controller import compare_striation_marks


class TestCompareStriationMarks:
    def test_raises_422_when_profiles_cannot_be_aligned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns 422 when marks have insufficient overlap to align."""
        monkeypatch.setattr(
            "processors.controller.correlate_striation_marks",
            lambda **kwargs: None,
        )
        dummy_mark = Mock(spec=Mark)
        dummy_profile = Mock(spec=Profile)

        with pytest.raises(HTTPException) as exc_info:
            compare_striation_marks(
                mark_ref=dummy_mark,
                mark_comp=dummy_mark,
                profile_ref=dummy_profile,
                profile_comp=dummy_profile,
            )

        assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
