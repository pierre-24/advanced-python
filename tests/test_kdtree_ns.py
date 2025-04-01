import numpy as np

from neighbour_search.kdtree_ns import KDTreeNeighbourSearch as NeighbourSearch


def test_knn_all_point_ok(five_close_points):
    ns = NeighbourSearch(five_close_points)

    assert set(ns.knn_search(0, 4)) == {1, 2, 3, 4}


def test_knn_some_point_ok(five_close_ten_apart_points):
    ns = NeighbourSearch(five_close_ten_apart_points)

    assert set(ns.knn_search(0, 4)) == {1, 2, 3, 4}


def test_ball_some_point_ok(five_close_points):
    ns = NeighbourSearch(five_close_points)

    assert set(ns.ball_search(np.array([0, 0]), np.sqrt(2))) == {0, 1, 2, 3, 4}


def test_ball_all_point_ok(five_close_ten_apart_points):
    ns = NeighbourSearch(five_close_ten_apart_points)

    assert set(ns.ball_search(np.array([0, 0]), np.sqrt(2))) == {0, 1, 2, 3, 4}
