import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np


def remove_duplicate(v: np.ndarray, f: np.ndarray):
    unique_v, v_inverse = np.unique(v, axis=0, return_inverse=True)
    if len(unique_v) != len(v):
        unique_f = np.unique(v_inverse[f], axis=0)
        return unique_v, unique_f
    else:
        return v, np.unique(f, axis=0)


def plot(vertices, faces, init1=None, init2=None, size=None):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(projection='3d')
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(vertices[faces])
    poly.set_edgecolor('black')
    poly.set_linewidth(0.25)

    ax.add_collection3d(poly)
    ax.scatter(*vertices.T, s=1)

    ax.view_init(init1, init2)

