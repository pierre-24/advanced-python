import numpy as np

from neighbour_search import AbstractNeighbourSearch, PriorityQueue


class NaiveNeighbourSearch(AbstractNeighbourSearch):
    """
    A "naive" implementation.

    `knn_search` is $\\mathcal{O}(n\\log(n))$, `ball_search` is $\\mathcal{O}(n)$.
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
            queue.put(i, d)

        # get the `k` first ones
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
