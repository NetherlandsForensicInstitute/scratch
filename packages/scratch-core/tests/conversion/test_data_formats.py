from conversion.data_formats import MarkStriation, MarkImpression


class TestMarkType:
    def test_impression_marks_are_identified_as_impressions(self):
        assert MarkImpression.BREECH_FACE_IMPRESSION.is_impression()
        assert MarkImpression.FIRING_PIN_IMPRESSION.is_impression()
        assert not MarkStriation.BULLET_GEA_STRIATION.is_impression()

    def test_striation_marks_are_identified_as_striations(self):
        assert MarkStriation.BULLET_GEA_STRIATION.is_striation()
        assert MarkStriation.FIRING_PIN_DRAG_STRIATION.is_striation()
        assert not MarkImpression.BREECH_FACE_IMPRESSION.is_striation()
