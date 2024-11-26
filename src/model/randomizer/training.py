import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_scatter import scatter_add
from functools import partial
from src.model.randomizer.utils import sample_multiple_graphs, accuracy
from src.model.randomizer.prediction import predict_smooth_pytorch
from src.model import general_utils
import time
import logging
import wandb


def get_time():
    torch.cuda.synchronize()
    return time.time()


def smooth_logits_pytorch(data, model, sample_config, sample_fn):
    n_samples = sample_config.get('n_samples', 1)

    logits = []
    for _ in range(n_samples):
        data_perturbed = sample_fn(data, sample_config)

        x_list = []
        adj_list = []
        for i in range(data_perturbed.num_graphs):
            # Extract node features for the i-th graph
            x_i = data_perturbed.x[data_perturbed.batch == i]
            x_list.append(x_i.cpu())

        adjs = utils.to_dense_adj(data_perturbed.edge_index, batch=data.batch)
        adj_list = [adjs[i].cpu().numpy() for i in range(adjs.size(0))]
        p_data = general_utils.get_data(x_list, adj_list, [0], data.batch)
        logits.append(model(p_data))
    return torch.stack(logits).mean(0)


def run_epoch_pytorch(
        model, optimizer, dataloader, nsamples, train, data_tuple=True, device='cuda'):
    """
    Run one epoch of training or evaluation.

    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        dataloader: Dataloader providing the data to run our model on
        nsamples: Number of samples over which the dataloader iterates
        train: Whether this epoch is used for training or evaluation
        data_tuple: Whether dataloader returns a tuple (x, y)
        device: Target device for computation

    Returns:
        Loss and accuracy in this epoch.
    """
    start = get_time()

    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over data
    for data in dataloader:
        if data_tuple:
            xb, yb = data[0].to(device), data[1].to(device)
        else:
            data.to(device)
            xb = data
            yb = data.y

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            # print(f'pred {pred}, yb {yb}')
            pred.to(device)
            yb.to(device)
            # print(f"pred {pred.device}, yb {yb.device}, {device}")
            loss = F.cross_entropy(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()

    epoch_loss /= nsamples
    epoch_acc /= nsamples
    epoch_time = get_time() - start
    return epoch_loss, epoch_acc, epoch_time


def train_pytorch(
        model, dataloaders, optimizer, lr_scheduler, n_samples,
        lr, weight_decay, patience, max_epochs, data_tuple=True,
        sample_fn=None, sample_config=None):
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5)

    if sample_config is not None:
        assert sample_fn is not None
        model_partial = partial(smooth_logits_pytorch,
                                model=model, sample_config=sample_config,
                                sample_fn=sample_fn)
    else:
        model_partial = partial(model)
    print(f"model on {model.device}")
    trace_val = []
    best_loss = np.inf
    for epoch in range(max_epochs):
        model.train()
        train_loss, train_acc, train_time = run_epoch_pytorch(
            model_partial, optimizer, dataloaders['train'], n_samples['train'],
            train=True, data_tuple=data_tuple, device=model.device)
        # wandb.log({f"Epoch {epoch + 1: >3}/{max_epochs}, "
        #              f"train loss: {train_loss:.2e}, "
        #              f"accuracy: {train_acc * 100:.2f}% ({train_time:.2f}s)"})
        logging.info(f"Epoch {epoch + 1: >3}/{max_epochs}, "
                     f"train loss: {train_loss:.2e}, "
                     f"accuracy: {train_acc * 100:.2f}% ({train_time:.2f}s)")

        model.eval()
        val_loss, val_acc, val_time = run_epoch_pytorch(
            model_partial, None, dataloaders['val'], n_samples['val'],
            train=False, data_tuple=data_tuple)
        trace_val.append(val_loss)
        logging.info(f"Epoch {epoch + 1: >3}/{max_epochs}, "
                     f"val loss: {val_loss:.2e}, "
                     f"accuracy: {val_acc * 100:.2f}% ({val_time:.2f}s)")

        lr_scheduler.step()

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

        # Early stopping
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return trace_val
