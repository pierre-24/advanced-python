"""
Quick and dirty script to get some timings.
"""

import numpy as np
import timeit

from neighbour_search import AbstractNeighbourSearch
from neighbour_search.naive_ns import NaiveNeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNeighbourSearch
from neighbour_search.balltree_ns import BallTreeNeighbourSearch


def time_ns(ns: AbstractNeighbourSearch, N: int = 10) -> tuple[float, float]:
    timer = timeit.Timer(lambda: ns.ball_search(np.random.random(2), 1))
    rp = timer.repeat(10, N)
    return np.mean(rp), np.std(rp)

for i in np.linspace(2, 6, 8):
    N = int(10 ** i)
    points = 100 * np.random.random((N, 2))

    ns_naive = NaiveNeighbourSearch(points)
    ns_kd = KDTreeNeighbourSearch(points, leaf_size=100)
    ns_ball = BallTreeNeighbourSearch(points, leaf_size=100)
    print(N, time_ns(ns_naive), time_ns(ns_kd), time_ns(ns_ball))
