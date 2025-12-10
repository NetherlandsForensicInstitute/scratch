from conversion.data_formats import MarkType


class TestMarkType:
    def test_impression_marks_are_identified_as_impressions(self):
        assert MarkType.BREECH_FACE_IMPRESSION.is_impression()
        assert MarkType.FIRING_PIN_IMPRESSION.is_impression()
        assert not MarkType.BULLET_GEA_STRIATION.is_impression()

    def test_striation_marks_are_identified_as_striations(self):
        assert MarkType.BULLET_GEA_STRIATION.is_striation()
        assert MarkType.FIRING_PIN_DRAG_STRIATION.is_striation()
        assert not MarkType.BREECH_FACE_IMPRESSION.is_striation()

    def test_all_marks_are_either_impression_or_striation(self):
        for mark in MarkType:
            assert mark.is_impression() ^ mark.is_striation(), (
                f"{mark} is neither or both"
            )
