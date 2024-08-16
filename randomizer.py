from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))

    # default params
    dataset = 'mutag'
    n_per_class = 20
    seed = 42

    patience = 50
    max_epochs = 1000
    lr = 1e-6
    weight_decay = 1e-5

    arch = 'GIN'
    n_hidden = 64
    p_dropout = 0.5

    pf_plus_adj = 0.0
    pf_minus_adj = 0.0

    pf_plus_att = 0.0
    pf_minus_att = 0.01

    n_samples_train = 1
    batch_size_train = 1

    n_samples_pre_eval = 10
    n_samples_eval = 100
    batch_size_eval = 10

    mean_softmax = False
    conf_alpha = 0.01
    early_stopping = True

    save_dir = 'data'


@ex.automain
def run(_config, dataset, n_per_class, seed,
        patience, max_epochs, lr, weight_decay, arch, n_hidden, p_dropout,
        pf_plus_adj, pf_plus_att, pf_minus_adj, pf_minus_att, conf_alpha,
        n_samples_train, n_samples_pre_eval, n_samples_eval, mean_softmax, early_stopping,
        batch_size_train, batch_size_eval, save_dir,
        ):
    import numpy as np
    import torch
    from src.model.classifier import GCN, GIN
    from src.datasets.graph_data_reader import DataReader, GraphData
    from src.model.randomizer.training import train_pytorch
    from src.model.randomizer.prediction import predict_smooth_pytorch
    from src.model.randomizer.cert import binary_certificate, joint_binary_certificate, minimize
    from src.model.randomizer.utils import (sample_batch_pyg)
    from src.model.general_utils import save_cetrificate
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    print(_config)

    if dataset.lower() not in ['mnist', 'mutag', 'proteins']:
        batch_size_train = min(batch_size_train, n_samples_train)
        batch_size_eval = min(batch_size_eval, n_samples_eval)

    sample_config = {
        'n_samples': n_samples_train,
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }

    # if we need to sample at least once and at least one flip probability is non-zero
    if n_samples_train > 0 and (pf_plus_adj+pf_plus_att+pf_minus_adj+pf_minus_att > 0):
        sample_config_train = sample_config
        sample_config_train['mean_softmax'] = mean_softmax
    else:
        sample_config_train = None
    sample_config_eval = sample_config.copy()
    sample_config_eval['n_samples'] = n_samples_eval
    
    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = n_samples_pre_eval
    
    pyg_dataset = TUDataset(
        root=f'data/datacache/{dataset.lower()}', name=dataset.upper())
    pyg_dataset.data.edge_attr = None
    # Caution: Degrees as features if pyg_dataset.x is None.
    n_graphs = {'train': int(0.8 * len(pyg_dataset)),
                'test': int(np.ceil(0.1 * len(pyg_dataset)))}
    n_graphs['val'] = len(pyg_dataset) - \
        n_graphs['train'] - n_graphs['test']
    
    dataloaders = {}
    dataloaders['train'] = PyGDataLoader(
        pyg_dataset[:n_graphs['train']], batch_size_train, shuffle=True)
    dataloaders['val'] = PyGDataLoader(pyg_dataset[n_graphs['train']:n_graphs['train'] + n_graphs['val']],
                                       batch_size_train, shuffle=False)
    dataloaders['test'] = PyGDataLoader(pyg_dataset[n_graphs['train'] + n_graphs['val']:],
                                        batch_size_train, shuffle=False)
    idx = {}
    idx['train'] = np.arange(n_graphs['train'])
    idx['val'] = np.arange(n_graphs['val'])
    idx['val'] += idx['train'][-1] + 1
    idx['test'] = np.arange(n_graphs['test'])
    idx['test'] += idx['val'][-1] + 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch.lower() == 'gin':
        model = GIN(n_feat=pyg_dataset.num_features,
                    n_class=pyg_dataset.num_classes,
                    n_layer=2,
                    agg_hidden=64,
                    fc_hidden=128,
                    dropout=0,
                    readout='avg',
                    device=device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5)
    
    trace_val = train_pytorch(
        model, dataloaders, optimizer, lr_scheduler, n_graphs,
        lr, weight_decay, patience, max_epochs,
        data_tuple=False, sample_fn=sample_batch_pyg,
        sample_config=sample_config_train)
    
    votes_dict = {}
    acc_majority = {}
    for split_name in ['test']:
        votes_dict[split_name], acc_majority[split_name] = predict_smooth_pytorch(
            model, dataloaders[split_name], n_graphs[split_name],
            n_classes=pyg_dataset.num_classes,
            data_tuple=False, sample_fn=sample_batch_pyg,
            sample_config=sample_config_eval)
    
    votes = votes_dict['test']
    votes_max = votes_dict['test'].argmax(1)
    
    pre_votes_dict = {}
    acc_clean = {}
    for split_name in ['test']:
        pre_votes_dict[split_name], acc_clean[split_name] = predict_smooth_pytorch(
            model, dataloaders[split_name], n_graphs[split_name],
            n_classes=pyg_dataset.num_classes,
            data_tuple=False, sample_fn=sample_batch_pyg,
            sample_config=sample_config_pre_eval)
    
    pre_votes = pre_votes_dict['test']
    pre_votes_max = votes_dict['test'].argmax(1)

    label = pyg_dataset.data.y[idx['test']]
    correct = votes_max == np.array(pyg_dataset.data.y[idx['test']])

    # we are perturbing ONLY the ATTRIBUTES
    if pf_plus_adj == 0 and pf_minus_adj == 0:
        print('Just ATT')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    # we are perturbing ONLY the GRAPH
    elif pf_plus_att == 0 and pf_minus_att == 0:
        print('Just ADJ')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)
    else:
        grid_base, grid_lower, grid_upper = joint_binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
            pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att)
    # mean_max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1).mean()
    # mean_max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1).mean()
    # mean_max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1).mean()
    # mean_max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1).mean()
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    
    # torch.save(model.state_dict(), save_name)
    # print(f'Saved model to {save_name}')
    test_idx = np.arange(n_graphs['test'])
    binary_class_cert = (grid_base > 0.5)[test_idx].T
    multi_class_cert = (grid_lower > grid_upper)[test_idx].T
    # the returned result will be written into the database
    results = {
        'clean_acc': acc_clean['test'],
        'majority_acc': acc_majority['test'],
        'correct': correct.tolist(),
        "binary": {
            "ratios": minimize(binary_class_cert.mean(-1).T),
            "cert_acc": minimize((correct * binary_class_cert).mean(-1).T)
        },
        "multiclass": {
            "ratios": minimize(multi_class_cert.mean(-1).T),
            "cert_acc": minimize((correct * multi_class_cert).mean(-1).T)
        }
    }
    # results = {
    #     'clean_acc': acc_clean['test'],
    #     'majority_acc': acc_majority['test']
    # }
    
    hparams = {
        'classifier': model.__class__.__name__.lower(),
        'smoothing_config': {
            'p': 1,
            'p_plus_adj': pf_plus_adj,
            'p_plus': pf_plus_att,
            'p_minus_adj': pf_minus_adj,
            'p_minus': pf_minus_att,
        },
    }
    save_cetrificate(results, dataset, hparams, f"{save_dir}/{hparams['classifier']}_{dataset}")
    import json
    with open(f'{dataset}_res.txt', 'a') as f:
        json.dump(sample_config, f, indent=4)
        json.dump({k: results[k] for k in ['clean_acc', 'majority_acc']}, f, indent=4)
    return {k: results[k] for k in ['clean_acc', 'majority_acc']}
