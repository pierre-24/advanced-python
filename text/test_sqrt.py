import pytest
import math


def _sqrt(x: float, tol: float = 1e-5) -> float:
    """Computes the square root of a number `x` up to a given convergence `tol`, using the Newton algorithm.

    :math:`x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}`, where :math:`f(x)=x^2-a`.

    Arguments:
        x: a floating point number
    Returns:
        the square root of `x`
    Raises:
        ValueError: if `x` is not a number
    """
    if x < 0:
        raise ValueError("x must be positive")
    a = 1
    while abs(a ** 2 - x) > tol:
        a = .5 * (a + x / a)

    return a


def test_sqrt():
    for i in [1, 5, 10, 100, 500]:
        assert math.sqrt(i) == pytest.approx(_sqrt(i), abs=1e-5)