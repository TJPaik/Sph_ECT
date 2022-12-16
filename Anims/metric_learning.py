# %%
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
import torch
from icosphere import icosphere
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Anims.gnn_model import Model

# %%
directions = torch.asarray([el for el in icosphere(7)][0])
ec_data = torch.from_numpy(np.load(f'Anims/euler_curves/original.npy'))
ec_data_random = [torch.from_numpy(np.load(f'Anims/euler_curves/random_{i}.npy')) for i in tqdm(range(4))]
ec_data_random_flip = [torch.from_numpy(np.load(f'Anims/euler_curves/random_{i}_flip.npy')) for i in tqdm(range(4))]
# %%
label = np.load('Anims/label.npy')
label_unique = np.unique(label).tolist()
label_n = np.asarray([label_unique.index(el) for el in label])
# %%
direction_dm_mat = torch.cdist(directions, directions) + torch.eye(len(directions)) * 2
edge_indices = torch.stack(torch.where(direction_dm_mat < 0.2))
distance_matrix = torch.from_numpy(np.load('Anims/pagerank_dist/dm0.npy')).cuda()
distance_matrix = distance_matrix / distance_matrix.max()
# %%
for i in range(len(directions)):
    assert len(torch.where(edge_indices[0] == i)[0]) == len(torch.where(edge_indices[1] == i)[0])

# %%
model = Model()
model.cuda()
# %%
train_idx = []
for el in ['face', 'head', 'horse', 'elephant', 'camel', 'cat', 'lion']:
    train_idx.extend(np.random.choice(np.where(label == el)[0], 2))
train_idx = torch.asarray(train_idx)
print(train_idx)

# Normalize
normalizing_factor = ec_data[train_idx].std()
ec_data /= normalizing_factor
# %%
ds = TensorDataset(train_idx)
dl = DataLoader(ds, 8, True)
ds_total = TensorDataset(torch.arange(len(ec_data)))
dl_total = DataLoader(ds_total, 8, False)
# %%
# Train
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.05)
model.train()
pbar = tqdm(range(400))
for epoch in pbar:
    loss_sum = 0
    for idx, in dl:
        optimizer.zero_grad()
        output = model(ec_data[idx].float().cuda(), edge_indices.cuda())
        sub_mat = distance_matrix[idx][:, idx].float()
        loss = nn.MSELoss()(
            torch.cdist(output, output),
            sub_mat
        )
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
        schedular.step()
    pbar.set_postfix_str(f'{loss_sum}')
# %%
####################################################################################################
# Eval
# training test
model.eval()
with torch.no_grad():
    outputs = []
    for idx, in tqdm(dl_total, total=len(dl_total)):
        outputs.append(
            model(ec_data[idx].float().cuda(), edge_indices.cuda()).cpu().detach()
        )
final_output = torch.cat(outputs).numpy()

plt.figure(figsize=(5, 5))
for el in label_unique:
    plt.scatter(*final_output[label == el].T, label=el, s=20, alpha=0.8)
plt.legend()
plt.tight_layout()
# plt.savefig('results/semi_result_3.png', dpi=300)
plt.show()
plt.close()
# %%
# Random Rotation / translation / flip
random_output = []
for r_data in chain(ec_data_random, ec_data_random_flip):
    model.eval()
    with torch.no_grad():
        outputs = []
        for idx, in tqdm(dl_total, total=len(dl_total)):
            outputs.append(
                model(r_data[idx].float().cuda() / normalizing_factor, edge_indices.cuda()).cpu().detach()
            )
    random_output.append(
        torch.cat(outputs).numpy()
    )

# %%
fig, axs = plt.subplots(4, 2, figsize=(10, 20))
for j in range(len(random_output)):
    for el in label_unique:
        axs[j // 2][j % 2].scatter(*random_output[j][label == el].T, label=el, s=20, alpha=0.8)
plt.legend()
plt.tight_layout()
# plt.savefig('results/semi_result_r3.png', dpi=300)
plt.show()
plt.close()
# %%
