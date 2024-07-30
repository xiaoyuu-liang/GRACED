import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from . import utils

class MPNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) model.
    """
    def __init__(self, input_dims: dict, n_mlp_layers: int, hidden_mlp_dims: dict, mlp_dropout: float,
                 n_gnn_layers: int, hidden_gnn_dims: dict, gnn_dropout: float, output_dims: dict,
                 act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()

        self.attr_predictor = MLP(input_dims, n_mlp_layers, hidden_mlp_dims, output_dims, mlp_dropout, act_fn_in, act_fn_out)

        self.link_predictor = GNN(input_dims, n_gnn_layers, hidden_gnn_dims, output_dims, gnn_dropout, act_fn_in, act_fn_out)
    
    def forward(self, X, E, y, node_mask):
        bs, n, bx, bx_c = X.shape

        t_X = y[:,0].unsqueeze(1)
        t_E = y[:,1].unsqueeze(1)

        X = self.attr_predictor(X, t_X, node_mask)                 # (bs, n, bx*bx_c)
        X = X.view(bs, n, bx, bx_c)                                 # (bs, n, bx, bx_c)

        E = self.link_predictor(X, E, t_E, node_mask)              # (bs, n, n, be)

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) model.
    """
    def __init__(self, input_dims: dict, n_layers: int, hidden_dims: dict, output_dims: dict,
                 dropout: float, act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()
        
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X']*input_dims['Xc'], hidden_dims['X']), act_fn_in,
                                      nn.Linear(hidden_dims['X'], hidden_dims['X']), act_fn_in)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_dims['y']), act_fn_in,
                                      nn.Linear(hidden_dims['y'], hidden_dims['y']), act_fn_in)

        self.mlp_layers = nn.ModuleList([MLPLayer(hidden_dims, act_fn_in, dropout) for _ in range(n_layers)])

        hidden_cat = (n_layers + 1) * (hidden_dims['X']) + hidden_dims['y']
        self.mlp_out = nn.Sequential(nn.Linear(hidden_cat, hidden_cat), act_fn_out,
                                    nn.Linear(hidden_cat, output_dims['X']*output_dims['Xc']))

    def forward(self, X, y, node_mask):
        bs, n, bx, bx_c = X.shape
        X = X.view(bs, n, -1)                   # (bs, n, bx*bx_c)
        
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1 
        X = self.mlp_in_X(X) * x_mask
        y = self.mlp_in_y(y)
        _, hy = y.shape
        
        X_list = [X]
        for layer in self.mlp_layers:
            X = layer(X, y)
            X = X * x_mask
            X_list.append(X)
        
        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat(X_list + [y_expand], dim=-1)

        X = self.mlp_out(X)

        return X


class MLPLayer(nn.Module):
    """
    Multi-layer Perceptron (MLP) layer.
    """
    def __init__(self, hidden_dims: dict, act_fn: nn.ReLU, dropout: float):
        super().__init__()
        
        self.update_X = nn.Sequential(nn.Linear(hidden_dims['X'] + hidden_dims['y'], hidden_dims['X']), act_fn,
                                      nn.LayerNorm(hidden_dims['X']), nn.Dropout(dropout))
        
    def forward(self, X, y):
        bs, n, hx = X.shape
        _, hy = y.shape

        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat([X, y_expand], dim=-1)

        X = self.update_X(X)                    # (bs, n, hx)
        return X


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) model.
    """
    def __init__(self, input_dims: dict, n_layers: int, hidden_dims: dict, output_dims: dict,
                 dropout: float, act_fn_in: nn.ReLU, act_fn_out: nn.ReLU):
        super().__init__()

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X']*input_dims['Xc'], hidden_dims['X']), act_fn_in,
                                      nn.Linear(hidden_dims['X'], hidden_dims['X']), act_fn_in)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_dims['y']), act_fn_in,
                                      nn.Linear(hidden_dims['y'], hidden_dims['y']), act_fn_in)

        self.gnn_layers = nn.ModuleList([GNNLayer(hidden_dims, act_fn_in, dropout) for _ in range(n_layers)])

        hidden_cat = (n_layers + 1) * (hidden_dims['X']) + hidden_dims['y']
        self.gnn_out = nn.Sequential(nn.Linear(hidden_cat, hidden_cat), act_fn_out,
                                    nn.Linear(hidden_cat, hidden_dims['E']))
        
        self.mlp_out = nn.Sequential(nn.Linear(hidden_dims['E'], hidden_dims['E']), act_fn_out,
                                     nn.Linear(hidden_dims['E'], output_dims['E']))
    
    def forward(self, X, E, y, node_mask):
        bs, n, bx, bx_c = X.shape
        X = X.view(bs, n, -1)                   # (bs, n, bx*bx_c)
        
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1 
        X = self.mlp_in_X(X) * x_mask

        y = self.mlp_in_y(y)
        _, hy = y.shape

        X_list = [X]
        for layer in self.gnn_layers:
            X = layer(X, E, y)
            X = X * x_mask
            X_list.append(X)
        
        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat(X_list + [y_expand], dim=-1)

        X = self.gnn_out(X)         # (bs, n, he)
        _, _, he = X.shape
        del X_list, y_expand

        # get edge features
        batch = torch.arange(bs, device=E.device).view(-1, 1).repeat(1, n).view(-1)

        adj = E[..., 1]
        adj_list = [adj[i] for i in range(bs)]
        adj_block_diag = torch.block_diag(*adj_list)
        edge_index = torch.nonzero(adj_block_diag).t()
        src, dst = edge_index
        del adj, adj_list, adj_block_diag

        stack_X = X.view(bs*n, he)
        edge_attr = stack_X[src] * stack_X[dst]         # (|E|, he)
        E = utils.to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        del edge_index, src, dst, stack_X, edge_attr, batch
        
        diag_mask = ~torch.eye(E.size(1), device=E.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        E = self.mlp_out(E)
        E = (E + E.transpose(1, 2)) / 2
        E = E * diag_mask

        return E


class GNNLayer(nn.Module):
    """
    Graph Neural Network (GNN) layer.
    """
    def __init__(self, hidden_dims: dict, act_fn: nn.ReLU, dropout: float):
        super().__init__()

        self.aggr_X = GCNConv(hidden_dims['X'], hidden_dims['X'])

        self.update_X = nn.Sequential(nn.Linear(hidden_dims['X'] + hidden_dims['y'], hidden_dims['X']), act_fn,
                                      nn.LayerNorm(hidden_dims['X']), nn.Dropout(dropout))
        
        
    def forward(self, X, E, y):
        bs, n, hx = X.shape
        _, hy = y.shape

        stack_X = X.view(bs*n, hx)

        adj = E[..., 1]
        adj_list = [adj[i] for i in range(bs)]
        adj_block_diag = torch.block_diag(*adj_list)
        edge_index = torch.nonzero(adj_block_diag).t()
        del adj, adj_list, adj_block_diag
    
        X = torch.cat([stack_X], dim=-1)
        X = self.aggr_X(X, edge_index)

        y_expand = y.unsqueeze(1).expand(bs, n, hy)
        X = torch.cat([X.view(bs, n, hx), y_expand], dim=-1)

        X = self.update_X(X)                                    # (bs, n, hx)
        
        return X