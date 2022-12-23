# %%
from synt import simp1

import numpy as np
from icosphere import icosphere
from lib.mesh2ec import mesh2ec
from lib.mesh import randomize_vertices

directions = icosphere(7)[0]
assert np.all(np.abs(np.linalg.norm(directions, axis=1) - 1) < 1e-3)
# %%
v, f = simp1()
upper_limit = 5
# %%
# Without random rotation & translation
resolutions = [16, 32, 64]
grids = [np.linspace(-upper_limit, upper_limit, el) for el in resolutions]
results = [mesh2ec(v, f, directions, grid) for grid in grids]
for el1, el2 in zip(resolutions, results):
    np.save(f'synt/ec_simp1_{el1}/original.npy', el2)

# %%
# With random rotation & translation
for i in range(4):
    print(i)
    results = [mesh2ec(randomize_vertices(v, upper_limit), f, directions, grid) for grid in grids]
    for el1, el2 in zip(resolutions, results):
        np.save(f'synt/ec_simp1_{el1}/random_{i}.npy', el2)
# %%
# With random rotation & translation with flip
for i in range(4):
    print(i)
    results = [
        mesh2ec(randomize_vertices(v, upper_limit) @ np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), f, directions,
                grid)
        for grid in grids]
    for el1, el2 in zip(resolutions, results):
        np.save(f'synt/ec_simp1_{el1}/random_flip_{i}.npy', el2)
# %%
