import os
import pathlib
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset

from .distribution import DistributionNodes
from .utils import to_dense, get_one_hot
from src.datasets.graph_data_reader import DataReader, GraphData


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 1,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]
    
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers)
    
    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(self.val_dataset, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers)
    
    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.test_dataset, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers)

    def node_counts(self, max_nodes_possible=1000):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types, node_attr):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.num_node_attr = len(node_attr)
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = to_dense(example_batch.x, example_batch.edge_index, 
                                       example_batch.edge_attr, example_batch.batch)

        self.input_dims = {'X': example_batch['x'].size(1),
                           'Xc': example_batch['x'].size(2),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1,     # + 1 due to time conditioning
                           'label': self.num_classes}      

        self.output_dims = {'X': example_batch['x'].size(1),
                            'Xc': example_batch['x'].size(2),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0,
                            'label': self.num_classes}



class AttributedGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None,):
        base_dir = pathlib.Path(os.path.abspath(__file__)).parents[3]
        self.dataset_name = dataset_name
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']
    
    def download(self):
        datareader = DataReader(data_dir=self.root)
        datareader.data['splits'] = datareader.data['splits'][0]      # del other folds
        num_graph = len(datareader.data['adj_list'])

        data_list = []
        for i in range(num_graph):
            attr_one_hot = get_one_hot(torch.LongTensor(datareader.data['features_onehot'][i]))
            edge_index = torch.LongTensor(datareader.data['edge_matrix_list'][i]).T
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            y = torch.zeros([1, 0]).float()
            num_nodes = datareader.data['node_count_list'][i] * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=attr_one_hot, # (N, F, 2)
                                             edge_index=edge_index, # 2 * |E| (sparse)
                                             edge_attr=edge_attr, # ｜E｜ * 2
                                             label=datareader.data['targets'][i],
                                             y=y,
                                             n_nodes=num_nodes)
            data_list.append(data)
        
        train_indices = datareader.data['splits']['train']
        idx = np.random.choice(len(train_indices), int(len(train_indices) * 0.1), replace=False)
        val_indices = train_indices[idx]
        train_indices = np.delete(train_indices, idx)
        test_indices = datareader.data['splits']['test']

        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for data in raw_dataset:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class AttributedGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=5000):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[3]
        self.root_path = os.path.join(base_path, self.datadir)[1:]

        datasets = {'train': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                    split='train', root=self.root_path),
                    'val': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                  split='val', root=self.root_path),
                    'test': AttributedGraphDataset(dataset_name=self.cfg.dataset.name,
                                                   split='test', root=self.root_path)}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class AttributedDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        datareader = DataReader(datamodule.root_path)

        datareader.data['splits'] = datareader.data['splits'][0]      # del other folds
        train_len = len(datareader.data['splits']['train'])
        val_len =  int(train_len * 0.1)
        train_len = train_len - val_len
        test_val = len(datareader.data['splits']['test'])
        self.split_len = {'train': train_len, 'val': val_len, 'test': test_val}

        self.node_counts = self.datamodule.node_counts() # node count marginal distribution

        node_attr = torch.Tensor(datareader.data['feature_margin'])
        self.node_attrs = torch.stack((1 - node_attr, node_attr), dim=1)        # node attributes marginal distribution

        node_types = torch.Tensor(datareader.data['label_margin'])
        self.node_types = torch.stack((1 - node_types, node_types), dim=1)      # node label marginal distribution

        edge_types = torch.Tensor(datareader.data['edge_margin'])
        self.edge_types =  torch.stack((1 - edge_types, edge_types), dim=1)[0]     # edge existence marginal distribution

        super().complete_infos(self.node_counts, self.node_types, self.node_attrs)