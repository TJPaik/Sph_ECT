from collections import Counter
from pathlib import Path

import numpy as np
from icosphere import icosphere
from joblib import Parallel, delayed
from scipy.stats import special_ortho_group
from tqdm import tqdm
from vedo import Mesh

mesh_data_folder = Path('/home/paiktj/Downloads/mesh_data')
obj_files = np.asarray([el.as_posix() for el in mesh_data_folder.glob('**/*.obj') if 'collapse' not in el.as_posix()])
obj_files = sorted(obj_files)
folder_names = np.asarray([el.split('/')[-2] for el in obj_files])
folder_names = np.asarray([el.split('-')[-2] for el in folder_names])
meshs = [Mesh(el) for el in tqdm(obj_files)]

vfs_original = [[
    np.unique(mesh.vertices(), axis=0),
    np.unique(np.asarray(mesh.faces()), axis=0)
] for mesh in tqdm(meshs)]

# Error cases
error_indices1 = []
for i, (v, f) in enumerate(vfs_original):
    if len(v) <= f.max():
        error_indices1.append(i)
print(Counter(folder_names))
print(Counter(folder_names[error_indices1]))

directions, _ = icosphere(7)
assert np.all(np.abs(np.linalg.norm(directions, axis=1) - 1) < 1e-3)

resolution = 1024 * 3
grid = np.linspace(-16, 16, resolution)


def get_euler_curve(v, f):
    try:
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
    except:
        return None


# %%
# Without random rotation & translation
results = Parallel(n_jobs=26, verbose=9)(
    delayed(get_euler_curve)(v, f) for i, (v, f) in enumerate(vfs_original) if i not in error_indices1)

error_indices2 = []
nresults = []
for i, el in enumerate(results):
    if el is None:
        error_indices2.append(i)
    else:
        nresults.append(el)
nresults = np.asarray(nresults)
assert len(error_indices2) == 0
np.save('Anims/euler_curves/original.npy', nresults)

# %%
# With random rotation & translation
for j in range(4):
    print('\t', j, end='')
    results = np.asarray(Parallel(n_jobs=26, verbose=9)(
        delayed(get_euler_curve)(v @ special_ortho_group.rvs(3), f)
        for i, (v, f) in enumerate(vfs_original) if i not in error_indices1))
    assert results.shape == nresults.shape
    np.save(f'Anims/euler_curves/random_{j}.npy', results)

# %%
# With random rotation & translation with flip
for j in range(4):
    print('\t', j, end='')
    results = np.asarray(Parallel(n_jobs=26, verbose=9)(
        delayed(get_euler_curve)(v @ np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ special_ortho_group.rvs(3), f)
        for i, (v, f) in enumerate(vfs_original) if i not in error_indices1))
    np.save(f'Anims/euler_curves/random_{j}_flip.npy', results)
