import numpy as np


def simp1():
    return (
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]).astype(float),
        np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    )
