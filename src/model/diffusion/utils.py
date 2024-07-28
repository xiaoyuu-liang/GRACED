import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb

import scipy.sparse as sp
import torch.nn.functional as F


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X      
        self.E = E      
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        e_mask = node_mask.unsqueeze(-1)
        x_mask = node_mask.unsqueeze(-1).unsqueeze(-1)          # bs, n, 1, 1
        e_mask1 = e_mask.unsqueeze(2)                           # bs, n, 1, 1
        e_mask2 = e_mask.unsqueeze(1)                           # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_jiont_diffuser_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')


def get_one_hot(graph):
    """
    Get the one-hot encoding of the graph.

    Parameters
    ----------
    graph: SparseGraph The graph to get the one-hot encoding from.

    Returns
    -------
    attr_one_hot: torch.Tensor of shape (N, F, 2) 
        The one-hot encoding of attr matrix.
        attr_one_hot[:, :, 0] = 1 <=> attr_matrix[i] = 0
        attr_one_hot[:, :, 1] = 1 <=> attr_matrix[i] = 1
    adj_one_hot: torch.Tensor of shape (N, N, 2)
        The one-hot encoding of adj matrix.
        adj_one_hot[:, :, 0] = 1 <=> adj_matrix[i, j] = 0
        adj_one_hot[:, :, 1] = 1 <=> adj_matrix[i, j] = 1
    """
    if sp.issparse(graph.attr_matrix):
        X = torch.LongTensor(graph.attr_matrix.todense())
    else:
        X = torch.LongTensor(graph.attr_matrix)
    
    if sp.issparse(graph.adj_matrix):
        A = torch.LongTensor(graph.adj_matrix.todense())
    else:
        A = torch.LongTensor(graph.adj_matrix)

    attr_one_hot_list = []
    for f in range(graph.num_node_attr):
        # (N, 2)
        attr_f_one_hot = F.one_hot(X[:, f], num_classes=2)
        attr_one_hot_list.append(attr_f_one_hot)
    # (F, N, 2)
    attr_one_hot = torch.stack(attr_one_hot_list, dim=0).float()
    # (N, F, 2)
    attr_one_hot = attr_one_hot.permute(1, 0, 2)

    # (N, N, 2)
    adj_one_hot = F.one_hot(A).float()

    return attr_one_hot, adj_one_hot

def get_marginal(graph):
    """
    Get the marginal distribution of the graph.

    Parameters
    ----------
    graph: SparseGraph The graph to get the marginal distribution from.

    Returns
    -------
    attr_margin: torch.Tensor of shape (F, 2) 
        The marginal distribution of X.
    label_margin: torch.Tensor of shape (|Y|) 
        The marginal distribution of Y.
    adj_margin: torch.Tensor of shape (2) 
        The marginal distribution of E.
    """
    attr_one_hot, adj_one_hot = get_one_hot(graph)
        
    # (F, 2)
    attr_one_hot = attr_one_hot.permute(1, 0, 2)
    attr_one_hot_count = attr_one_hot.sum(dim=1)
    attr_margin = attr_one_hot_count / attr_one_hot_count.sum(dim=1, keepdim=True)

    labels = graph.labels.copy()
    label_one_hot = F.one_hot(torch.LongTensor(labels), graph.num_classes)
    label_sum = label_one_hot.sum(dim=0)
    label_margin = label_sum / label_sum.sum()

    adj_one_hot_count = adj_one_hot.sum(dim=0).sum(dim=0)
    adj_margin = adj_one_hot_count / adj_one_hot_count.sum()


    return attr_margin, label_margin, adj_margin


