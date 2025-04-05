"""
Quick and dirty script to get some timings.
"""
import argparse

import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt

from neighbour_search import AbstractNeighbourSearch
from neighbour_search.naive_ns import NaiveNeighbourSearch
from neighbour_search.kdtree_ns import KDTreeNeighbourSearch
from neighbour_search.balltree_ns import BallTreeNeighbourSearch


def time_ball(ns: AbstractNeighbourSearch, N: int = 10, repeat: int = 25) -> tuple[np.floating, np.floating]:
    """
    Time (as mean and standard deviation) for `N` ball searches, repeated `repeat` times.
    """
    timer = timeit.Timer(lambda: ns.ball_search(np.sqrt(len(ns)) * np.random.random(2), len(ns) ** -.5 / 10))
    rp = timer.repeat(repeat, N)
    return np.mean(rp), np.std(rp)


def time_knn(ns: AbstractNeighbourSearch, N: int = 10, repeat: int = 25) -> tuple[np.floating, np.floating]:
    """
    Time (as mean and standard deviation) for `N` kNN, repeated `repeat` times.
    """
    timer = timeit.Timer(lambda: ns.knn_search(np.random.randint(0, len(ns)), 10))
    rp = timer.repeat(repeat, N)
    return np.mean(rp) / N, np.std(rp) / N


def compute_perfs(mm: tuple[int, int], n: int = 5):
    data_ball = []
    data_knn = []

    for j, i in enumerate(np.linspace(*mm, (mm[1] - mm[0]) * n)):
        N = int(10 ** i)

        print(j, i, N)

        points = np.sqrt(N) * np.random.random((N, 2))

        ns_naive = NaiveNeighbourSearch(points)
        ns_kd = KDTreeNeighbourSearch(points, leaf_size=100)
        ns_ball = BallTreeNeighbourSearch(points, leaf_size=100)

        data_ball.append(['ball', N, *time_ball(ns_naive), *time_ball(ns_kd), *time_ball(ns_ball)])
        data_knn.append(['knn', N, *time_knn(ns_naive), *time_knn(ns_kd), *time_knn(ns_ball)])

    return pd.DataFrame(
        data_ball + data_knn,
        columns=['test', 'N', 'naive_mean', 'naive_std', 'kdt_mean', 'kdt_std', 'ballt_mean', 'ballt_std']
    )


RANGE = 3, 7


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', type=int, default=RANGE[0], help='min')
    parser.add_argument('-M', type=int, default=RANGE[1], help='max')
    parser.add_argument('-t', type=int, default=5, help='subticks')
    parser.add_argument('-c', action='store_true', help='Compute perfs')
    parser.add_argument('-d', type=str, help='CSV file to store/read results', default='performances.csv')
    parser.add_argument('-o', type=str, help='image', default='performances.png')

    args = parser.parse_args()

    if args.c:
        data = compute_perfs((args.m, args.M), args.t)
        data.to_csv(args.d, index=False)
    else:
        data = pd.read_csv(args.d)

    figure = plt.figure(figsize=(6, 8))
    ax1, ax2 = figure.subplots(2, sharex=True, sharey=True)

    # ball test
    subdata = data[data['test'] == 'ball']
    ax1.errorbar(
        subdata['N'], subdata['naive_mean'], yerr=2 * subdata['naive_std'], fmt='o-', capsize=5, label='Naive')
    ax1.errorbar(
        subdata['N'], subdata['kdt_mean'], yerr=2 * subdata['kdt_std'], fmt='o-', capsize=5, label='KDTree')
    ax1.errorbar(
        subdata['N'], subdata['ballt_mean'], yerr=2 * subdata['ballt_std'], fmt='o-', capsize=5, label='BallTree')

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(subdata['N'].min(), subdata['N'].max())

    ax1.legend()
    ax1.set_ylabel('Mean time for selection (s)')
    ax1.set_title('Ball search')

    # knn test
    subdata = data[data['test'] == 'knn']
    ax2.errorbar(
        subdata['N'], subdata['naive_mean'], yerr=2 * subdata['naive_std'], fmt='o-', capsize=5, label='Naive')
    ax2.errorbar(
        subdata['N'], subdata['kdt_mean'], yerr=2 * subdata['kdt_std'], fmt='o-', capsize=5, label='KDTree')
    ax2.errorbar(
        subdata['N'], subdata['ballt_mean'], yerr=2 * subdata['ballt_std'], fmt='o-', capsize=5, label='BallTree')

    ax2.set_xlabel('N')
    ax2.set_ylabel('Mean time for selection (s)')
    ax2.set_title('kNN search')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(args.o, dpi=300)
