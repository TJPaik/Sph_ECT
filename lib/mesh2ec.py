from collections import Counter

import numpy as np


def mesh2ec(v: np.ndarray, f: np.ndarray, directions: np.ndarray, grid: np.ndarray):
    """
    :param v: shape == (N, 3). ex: [[0.1, 0.2, 0.3], ...]
    :param f: shape == (F, 3). ex: [[0, 1, 2], ...]
    :param directions: shape == (M, 3) with norm 1. ex: [[1, 0, 0], ...]
    :param grid: shape == (G, ). ex: [-16, -15, -14, ...]
    :return: Euler curve for each direction.
    """
    resolution = len(grid)
    e = set()
    for el in f:
        e.add(frozenset([el[0], el[1]]))
        e.add(frozenset([el[1], el[2]]))
        e.add(frozenset([el[0], el[2]]))
    e = np.asarray([list(el) for el in e])

    ips = v @ directions.T
    face_v_max = np.max(ips[f], axis=1).T
    edge_v_max = np.max(ips[e], axis=1).T
    ips = ips.T

    v_digit = np.digitize(ips, grid)
    e_digit = np.digitize(edge_v_max, grid)
    f_digit = np.digitize(face_v_max, grid)

    result = np.zeros((len(v_digit), resolution))

    for i in range(len(result)):
        for k, v in Counter(v_digit[i]).items():
            result[i][k] += v
    for i in range(len(result)):
        for k, v in Counter(e_digit[i]).items():
            result[i][k] -= v
    for i in range(len(result)):
        for k, v in Counter(f_digit[i]).items():
            result[i][k] += v

    result = np.cumsum(result, axis=1)
    return result
