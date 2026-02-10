import pytest
from container_models.base import Pair


class TestPairArithmetic:
    """Tests for Pair arithmetic operations."""

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(Pair(1, 2), Pair(3, 4), Pair(4, 6), id="pair+pair"),
            pytest.param(Pair(1, 2), 10, Pair(11, 12), id="pair+scalar"),
            pytest.param(10, Pair(1, 2), Pair(11, 12), id="scalar+pair"),
            pytest.param(Pair(5, 10), 0, Pair(5, 10), id="pair+zero"),
        ],
    )
    def test_add(self, left, right, expected) -> None:
        assert left + right == expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(Pair(5, 6), Pair(1, 2), Pair(4, 4), id="pair-pair"),
            pytest.param(Pair(5, 6), 1, Pair(4, 5), id="pair-scalar"),
            pytest.param(10, Pair(1, 2), Pair(9, 8), id="scalar-pair"),
            pytest.param(Pair(5, 10), Pair(5, 10), Pair(0, 0), id="pair-self"),
        ],
    )
    def test_sub(self, left, right, expected) -> None:
        assert left - right == expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(Pair(2, 3), Pair(4, 5), Pair(8, 15), id="pair*pair"),
            pytest.param(Pair(2, 3), 2, Pair(4, 6), id="pair*scalar"),
            pytest.param(2, Pair(3, 4), Pair(6, 8), id="scalar*pair"),
            pytest.param(Pair(5, 10), 1, Pair(5, 10), id="pair*one"),
            pytest.param(Pair(5, 10), 0, Pair(0, 0), id="pair*zero"),
        ],
    )
    def test_mul(self, left, right, expected) -> None:
        assert left * right == expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(Pair(10, 20), Pair(2, 4), Pair(5.0, 5.0), id="pair/pair"),
            pytest.param(Pair(10, 20), 2, Pair(5.0, 10.0), id="pair/scalar"),
            pytest.param(12, Pair(3, 4), Pair(4.0, 3.0), id="scalar/pair"),
            pytest.param(
                Pair(5.0, 10.0), Pair(5.0, 10.0), Pair(1.0, 1.0), id="pair/self"
            ),
        ],
    )
    def test_truediv(self, left, right, expected) -> None:
        assert left / right == expected

    def test_chained_operations(self) -> None:
        # (Pair(1, 2) + 3) * 2 = Pair(4, 5) * 2 = Pair(8, 10)
        assert (Pair(1, 2) + 3) * 2 == Pair(8, 10)

    def test_float_operations(self) -> None:
        assert Pair(1.5, 2.5) + Pair(0.5, 0.5) == Pair(2.0, 3.0)
        assert Pair(3.0, 4.0) * 0.5 == Pair(1.5, 2.0)


class TestPairMap:
    """Tests for Pair.map method."""

    @pytest.mark.parametrize(
        ("pair", "func", "other", "expected"),
        [
            pytest.param(
                Pair(1, 2), lambda x: x * 2, None, Pair(2, 4), id="single_arg"
            ),
            pytest.param(
                Pair(1, 2), lambda x, y: x + y, Pair(3, 4), Pair(4, 6), id="with_pair"
            ),
            pytest.param(
                Pair(1, 2), lambda x, y: x * y, [10, 20], Pair(10, 40), id="with_list"
            ),
        ],
    )
    def test_map(self, pair, func, other, expected) -> None:
        assert pair.map(func, other=other) == expected
