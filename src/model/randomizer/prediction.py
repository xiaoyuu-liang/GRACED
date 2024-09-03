import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from src.model.randomizer.utils import sample_multiple_graphs, binary_perturb
from tqdm.autonotebook import tqdm
from src.model import general_utils


def predict_smooth_pytorch(model, dataloader, n_data, n_classes,
                           data_tuple=True, sample_fn=None, sample_config=None):
    device = next(model.parameters()).device

    n_samples = sample_config.get('n_samples', 1)

    model.eval()
    ncorr = 0
    votes = torch.zeros((n_data, n_classes), dtype=torch.long, device=device)
    for ibatch, data in enumerate(dataloader):
        if data_tuple:
            xb, yb = data[0].to(device), data[1].to(device)
        else:
            data.to(device)
            xb = data
            yb = data.y

        batch_idx = ibatch * dataloader.batch_size
        for _ in tqdm(range(n_samples)):
            data_perturbed = sample_fn(xb, sample_config)
            x_list = []
            adj_list = []
            
            for i in range(data_perturbed.num_graphs):
                # Extract node features for the i-th graph
                x_i = data_perturbed.x[data_perturbed.batch == i]
                x_list.append(x_i.cpu())

            adjs = utils.to_dense_adj(data_perturbed.edge_index, batch=data.batch)
            adj_list = [adjs[i].cpu().numpy() for i in range(adjs.size(0))]
            p_data = general_utils.get_data(x_list, adj_list, [0], data.batch)

            preds = model(p_data).argmax(1)
            # print(preds)
            preds_onehot = F.one_hot(preds, n_classes)
            votes[batch_idx:batch_idx + yb.shape[0]] += preds_onehot
        ncorr += (votes[batch_idx:batch_idx + yb.shape[0]].argmax(1) == yb).sum().item()
    return votes.cpu().numpy(), ncorr / n_data
