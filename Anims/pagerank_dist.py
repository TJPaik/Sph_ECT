# %%
from collections import Counter
from pathlib import Path

import gudhi
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gudhi.simplex_tree import SimplexTree
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
from tqdm import tqdm
from vedo import Mesh

FACTOR = 10000
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


# %%
def get_persistence(v, f):
    st = SimplexTree(None)
    for _ in f:
        st.insert(_)
    assert st.num_vertices() == len(v)

    g = nx.Graph()
    for el in st.get_skeleton(1):
        if len(el[0]) == 1:
            g.add_node(el[0][0])
        elif len(el[0]) == 2:
            g.add_edge(*el[0])
        else:
            raise ValueError
    assert g.number_of_nodes() == len(v)
    pr = nx.pagerank(g, alpha=0.85)
    [st.assign_filtration([i], pr[i]) for i in range(len(v))]
    assert st.make_filtration_non_decreasing()
    assert not st.make_filtration_non_decreasing()
    result = st.persistence()
    return result


# %%
pss = Parallel(n_jobs=26, verbose=9) \
    (delayed(get_persistence)(el[0], el[1]) for i, el in enumerate(vfs_original) if i not in error_indices1)
pss0 = [FACTOR * np.asarray([el[1] for el in el2 if el[0] == 0]) for el2 in pss]
pss1 = [FACTOR * np.asarray([el[1] for el in el2 if el[0] == 1]) for el2 in pss]
# %%
dm0 = np.zeros((len(pss), len(pss)))
dm1 = np.zeros((len(pss), len(pss)))
for i in range(len(pss)):
    print(i)
    for j in tqdm(range(i + 1, len(pss))):
        dist = gudhi.bottleneck_distance(pss0[i], pss0[j])
        dm0[i][j], dm0[j][i] = dist, dist
        dist = gudhi.bottleneck_distance(pss1[i], pss1[j])
        dm1[i][j], dm1[j][i] = dist, dist
# %%
np.save('Anims/pagerank_dist/dm0.npy', dm0)
np.save('Anims/pagerank_dist/dm1.npy', dm1)
# %%
# %%
####################################################################################################
dm0 = np.load('Anims/pagerank_dist/dm0.npy')
dm1 = np.load('Anims/pagerank_dist/dm1.npy')
dm1[dm1 == np.inf] = 1
assert ~np.any(dm0 == np.inf)


# %%
def tsne_visualize(dm, label):
    tsne = TSNE(metric='precomputed')
    result_tsne0 = tsne.fit_transform(dm)
    label_unique = np.unique(label)
    for el in label_unique:
        _ = label == el
        plt.scatter(*result_tsne0[_].T, label=el)
    plt.legend()
    plt.show()
    return result_tsne0


# %%
plt.figure()
plt.title('dm_0')
tsne_visualize(dm0, np.asarray([el for i, el in enumerate(folder_names) if i not in error_indices1]))
# %%
plt.figure()
plt.title('dm_1')
tsne_visualize(dm1, np.asarray([el for i, el in enumerate(folder_names) if i not in error_indices1]))
# %%%
