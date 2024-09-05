import torch
import numpy as np
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as utils
from torch_geometric.transforms import OneHotDegree

dataset = 'IMDB-BINARY'
batch_size_train = 32

# Load the dataset
pyg_dataset = TUDataset(root=f'data/datacache/{dataset.lower()}', name=dataset.upper(), transform=OneHotDegree(max_degree=134))
# if dataset.lower() == 'imdb-binary':
#     max_degree = 136  # Set the maximum degree for one-hot encoding
#     print('Assigning degrees as features if data.x is None.')
    
#     for i, data in enumerate(pyg_dataset):
#         if data.x is None:
#             # Compute the node degrees
#             degrees = utils.degree(data.edge_index[0], data.num_nodes).to(torch.long)
#             # Convert the degrees to one-hot encoding
#             data.x = torch.nn.functional.one_hot(degrees, num_classes=max_degree).float()
#             assert hasattr(data, 'x'), f'Missing node features in graph {i}'
#             print(f"Graph {i} node features: {data.x}")

# Verify if the node features are correctly assigned
for i, data in enumerate(pyg_dataset):
    print(f"Graph {i} node features: {len(data.x[0])}")

# Split the dataset into training, validation, and test sets
n_graphs = {'train': int(0.8 * len(pyg_dataset)),
            'test': int(np.ceil(0.1 * len(pyg_dataset)))}
n_graphs['val'] = len(pyg_dataset) - n_graphs['train'] - n_graphs['test']

# Create dataloaders
from torch_geometric.loader import DataLoader as PyGDataLoader

dataloaders = {}
dataloaders['train'] = PyGDataLoader(pyg_dataset[:n_graphs['train']], batch_size_train, shuffle=True)
dataloaders['val'] = PyGDataLoader(pyg_dataset[n_graphs['train']:n_graphs['train'] + n_graphs['val']],
                                   batch_size_train, shuffle=False)
dataloaders['test'] = PyGDataLoader(pyg_dataset[n_graphs['train'] + n_graphs['val']:],
                                    batch_size_train, shuffle=False)

# Print node features for the first batch in the training dataloader
for data in dataloaders['train']:
    print(data.x)
    break

# Indices for training, validation, and test sets
idx = {}
idx['train'] = np.arange(n_graphs['train'])
idx['val'] = np.arange(n_graphs['val']) + idx['train'][-1] + 1