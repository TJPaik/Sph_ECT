from collections import Counter
from pathlib import Path

import numpy as np
from icosphere import icosphere
from joblib import Parallel, delayed
from scipy.stats import special_ortho_group
from tqdm import tqdm
from vedo import Mesh

from lib.mesh import remove_duplicate
from lib.mesh2ec import mesh2ec

mesh_data_folder = Path('/home/paiktj/Downloads/mesh_data')
obj_files = np.asarray([el.as_posix() for el in mesh_data_folder.glob('**/*.obj') if 'collapse' not in el.as_posix()])
obj_files = sorted(obj_files)
folder_names = np.asarray([el.split('/')[-2] for el in obj_files])
folder_names = np.asarray([el.split('-')[-2] for el in folder_names])
meshs = [Mesh(el) for el in tqdm(obj_files)]
vfs_original = [remove_duplicate(mesh.vertices(), mesh.faces()) for mesh in tqdm(meshs)]
print(Counter(folder_names))
# %%
directions = icosphere(7)[0]
assert np.all(np.abs(np.linalg.norm(directions, axis=1) - 1) < 1e-3)
# %%
# Without random rotation & translation
resolutions = [512, 1024, 3072]
grids = [np.linspace(-16, 16, el) for el in resolutions]
results = [np.asarray(Parallel(n_jobs=26, verbose=9)
                      (delayed(mesh2ec)(v, f, directions, grid)
                       for v, f in vfs_original)) for grid in grids]
for el1, el2 in zip(resolutions, results):
    np.save(f'Anims/euler_curves_{el1}/original.npy', el2)


# %%
def randomize_vertices(v):
    tmp = 16 - np.max(np.linalg.norm(v, axis=1))
    return v @ special_ortho_group.rvs(3) + (np.random.random(3) / np.sqrt(3)) * tmp


# With random rotation & translation
for i in range(4):
    print(i)
    results = [np.asarray(Parallel(n_jobs=26, verbose=9)(
        delayed(mesh2ec)(randomize_vertices(v), f, directions, grid)
        for v, f in vfs_original
    )) for grid in grids]
    for el1, el2 in zip(resolutions, results):
        np.save(f'Anims/euler_curves_{el1}/random_{i}.npy', el2)
# %%
# With random rotation & translation with flip
for i in range(4):
    print(i)
    results = [np.asarray(Parallel(n_jobs=26, verbose=9)(
        delayed(mesh2ec)(randomize_vertices(v) @ np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), f, directions, grid)
        for v, f in vfs_original
    )) for grid in grids]
    for el1, el2 in zip(resolutions, results):
        np.save(f'Anims/euler_curves_{el1}/random_{i}_flip.npy', el2)
# %%
