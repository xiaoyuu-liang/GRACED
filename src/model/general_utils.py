import torch
import networkx as nx
import numpy as np

def save_cetrificate(dict_to_save, dataset_config, hparams, path):

    if isinstance(dataset_config, str):
        dataset = dataset_config
    else:
        dataset = dataset_config['name']
    arch = hparams['classifier']
    p = hparams['smoothing_config']['p']
    p_plus = hparams['smoothing_config']['p_plus']
    p_minus = hparams['smoothing_config']['p_minus']
    p_plus_adj = hparams['smoothing_config']['p_plus_adj']
    p_minus_adj = hparams['smoothing_config']['p_minus_adj']

    print(f'saving to {path}/{arch}_{dataset}_[{p}-X{p_plus:.2f}-{p_minus:.2f}-E{p_plus_adj:.2f}-{p_minus_adj:.2f}].pth')
    torch.save(dict_to_save, f'{path}/{arch}_{dataset}_[{p}-X{p_plus:.2f}-{p_minus:.2f}-E{p_plus_adj:.2f}-{p_minus_adj:.2f}].pth')


def get_node_features_degree(adj_list):
    node_features_list = []
    for adj in adj_list:
        sub_list = []
        for feature in nx.from_numpy_matrix(np.array(adj)).degree():
            sub_list.append(feature[1])
        node_features_list.append(np.array(sub_list))
    return node_features_list


def get_max_neighbor(degree_list):
    max_neighbor_list = []
    
    for degrees in degree_list:
        max_neighbor_list.append(int(max(degrees)))
    return max_neighbor_list


def get_node_count_list(adj_list):
    node_count_list = []
    
    for adj in adj_list:
        node_count_list.append(len(adj))
                    
    return node_count_list

def get_edge_matrix_list(adj_list):
    edge_matrix_list = []
    max_edge_matrix = 0
    
    for adj in adj_list:
        edge_matrix = []
        for i in range(len(adj)):
            for j in range(len(adj[0])):
                if adj[i][j] == 1:
                    edge_matrix.append((i,j))
        if len(edge_matrix) > max_edge_matrix:
            max_edge_matrix = len(edge_matrix)
        edge_matrix_list.append(np.array(edge_matrix))
                    
    return edge_matrix_list, max_edge_matrix


def pad(mtx_list, desired_dim1, desired_dim2=None, value=0, mode='edge_matrix'):
    
    padded_mtx = np.array([
        np.pad(mtx, ((0, desired_dim1 - mtx.shape[0]), (0, desired_dim2 - mtx.shape[1])), 'constant', constant_values=0)
        for mtx in mtx_list
    ])
    
    return padded_mtx


def get_data(x_list, adj_list, label, node_mask):
    data = []
    n_feat = x_list[0].shape[1]
    N_node_max = max([len(x) for x in x_list])
    N_nodes = [adj_list[i].shape[0] for i in range(len(adj_list))]  

    data.append(torch.FloatTensor(pad(x_list, N_node_max, n_feat)))                     # x
    data.append(torch.FloatTensor(pad(adj_list, N_node_max, N_node_max)))               # adj
            
    data.append(node_mask)                                                              # node_mask

    data.append(torch.LongTensor([N_nodes]))                                            # N_nodes
    data.append(label)                                                                  # label
    max_neighbor_list = get_max_neighbor(get_node_features_degree(adj_list))    
    data.append(max_neighbor_list)                                                      # max_neighbor_list

    return data
