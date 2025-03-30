"""
Naive neighbour search implementation.

+ Creation is :math:`\\mathcal{O}(1)`.
+ It uses the ``PriorityQueue`` in ``knn_search()``, which is therefore :math:`\\mathcal{O}(n\\log(n))`.
+ It simply loops through all positions for ``ball_search()``, which is :math:`\\mathcal{O}(n)`.
"""

import numpy as np

from neighbour_search import AbstractNeighbourSearch, PriorityQueue


class NaiveNeighbourSearch(AbstractNeighbourSearch):
    """
    A "naive" implementation.

    Example:
        >>> import numpy as np
        >>> from neighbour_search.naive_ns import NaiveNeighbourSearch as NeighbourSearch
        >>> neighbour_search = NeighbourSearch(np.array([[0, 0], [1, 1]]))
        >>> neighbour_search.knn_search(0, 1)
        [1]
        >>> neighbour_search.ball_search(np.array([0, 0]), 5)
        [0, 1]
    """

    def knn_search(self, target: int, k: int) -> list[int]:
        # check preconditions
        assert 0 <= target < len(self)
        assert 0 <= k < len(self)

        # create queue
        queue = PriorityQueue()

        # put item in queue
        tp = self.positions[target]
        for i, p in enumerate(self.positions):
            if i == target:
                continue

            d = (p[0] - tp[0]) ** 2 + (p[1] - tp[1]) ** 2
            queue.push(i, d)

        # get the ``k`` first ones
        return [queue.pop() for _ in range(k)]

    def ball_search(self, position: np.ndarray[float], distance: float) -> list[int]:
        # check preconditions
        assert distance >= 0

        # create queue
        queue = []

        # put item in queue
        for i, p in enumerate(self.positions):
            d = (p[0] - position[0]) ** 2 + (p[1] - position[1]) ** 2
            if d <= distance ** 2:
                queue.append(i)

        # return the result
        return queue
