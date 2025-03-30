import pytest
import numpy as np


@pytest.fixture
def five_close_points():
    """Return 5 points that are all contained in [-1,1] x [-1,1].
    """

    center = np.random.random(2)

    return np.array([
        center + [.2, 0],
        center - [.2, 0],
        center + [0, .2],
        center + [-.2, .2],
        center + [.2, .2],
    ])


@pytest.fixture
def five_close_ten_apart_points(five_close_points):
    """Return 5 points that are all contained in [-1,1] x [-1,1], and 10 points that are not.
    """

    return np.vstack([five_close_points] + [5 * np.random.random(2) + [1, 1] for _ in range(10)])
