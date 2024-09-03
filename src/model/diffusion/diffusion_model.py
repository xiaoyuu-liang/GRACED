import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import pytorch_lightning as pl
import numpy as np
import time
import torch_geometric.data
import wandb
import os
from tqdm import tqdm

from torch_geometric.utils import to_dense_adj

from src.model.diffusion.train_metrics import TrainLossDiscrete, NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.model.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.model.diffusion.transformer_model import GraphTransformer
from src.model.diffusion.mpnn import MPNN
from src.model.diffusion import utils
from src.model.diffusion import diffusion_utils
from src.model.randomizer.cert import certify
from src.model import general_utils


class GraphDiffusionModel(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, extra_features, train_metrics):
        super().__init__()
        
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Xcdim = input_dims['Xc']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.labeldim = input_dims['label']
        self.Xdim_output = output_dims['X']
        self.Xcdim_output = output_dims['Xc']   
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.labeldim_output = output_dims['label']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.train_metrics = train_metrics

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        
        # self.model = GraphTransformer(n_layers=cfg.model.n_layers,
        #                               input_dims=input_dims,
        #                               hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        #                               hidden_dims=cfg.model.hidden_dims,
        #                               output_dims=output_dims,
        #                               act_fn_in=nn.ReLU(),
        #                               act_fn_out=nn.ReLU())
        self.model = MPNN(input_dims=input_dims,
                          n_mlp_layers=cfg.model.n_mlp_layers,
                          hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                          mlp_dropout=cfg.model.mlp_dropout,
                          n_gnn_layers=cfg.model.n_gnn_layers,
                          hidden_gnn_dims=cfg.model.hidden_gnn_dims,
                          gnn_dropout=cfg.model.gnn_dropout,
                          output_dims=output_dims,
                          act_fn_in=nn.ReLU(),
                          act_fn_out=nn.ReLU())
        
        if cfg.model.transition == 'marginal':
            x_marginals = self.dataset_info.node_attrs
            e_marginals = self.dataset_info.edge_types

            print(f"""Marginal distribution of the classes: 
                  {np.vectorize(lambda x: "{:.4f}".format(x))(x_marginals[:10])} for nodes, 
                  {np.vectorize(lambda x: "{:.4f}".format(x))(e_marginals)} for edges""")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)
            
        self.save_hyperparameters(ignore=['train_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def configure_optimizers(self):
        print('Using AdamW optimizer')
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                      weight_decay=self.cfg.train.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
        #                                                        factor=self.cfg.train.shceduler_factor, 
        #                                                        patience=self.cfg.train.shceduler_patience)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train.n_epochs, verbose=True)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/epoch_NLL"}
        return optimizer
    
    def on_fit_start(self) -> None:
        print("on fit starting")
        # self.train_iterations = len(self.trainer.train_dataloader)
        print(f"Size of the input features:")
        print(f"node_attr={self.Xdim}, node_attr_classes={self.Xcdim}, edge_classes={self.Edim}, time_condition={self.ydim}, node_classes={self.labeldim}")
        print(f"Size of the output features")
        print(f"node_attr={self.Xdim_output}, node_attr_classes={self.Xcdim_output}, edge_classes={self.Edim_output}, time_condition={self.ydim_output}, node_classes={self.labeldim_output}")
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
    
    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)
        
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)
        
        return {'loss': loss}
    
    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        # epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        # self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
    
    @torch.no_grad()
    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        # data_labels = torch.full((dense_data.X.size(0), dense_data.X.size(1)), -1, dtype=torch.long, device=dense_data.X.device)
        # for i, label in enumerate(data.labels):
        #     data_labels[i, :len(label)] = torch.LongTensor(label)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)

        return {'loss': nll}
    
    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Node type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        self.print("Done validating.")
    
    @torch.no_grad()
    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        # data_labels = torch.full((dense_data.X.size(0), dense_data.X.size(1)), -1, dtype=torch.long, device=dense_data.X.device)
        # for i, label in enumerate(data.labels):
        #     data_labels[i, :len(label)] = torch.LongTensor(label)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)

        return {'loss': nll}
    
    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        torch.cuda.empty_cache()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
    
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Node type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')
        self.print("Done testing.")

    def apply_noise(self, X, E, y, node_mask, t_X=None, t_E=None):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        if t_X is not None and t_E is not None:
            t_X_int = torch.full((X.size(0), 1), t_X, device=X.device).float()              #(bs, 1)
            s_X_int = t_X_int - 1
            t_E_int = torch.full((X.size(0), 1), t_E, device=X.device).float()
            s_E_int = t_E_int - 1
        elif not t_X and not t_E:
            lowest_t = 0 if self.training else 1
            t_X_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
            s_X_int = t_X_int - 1
            t_E_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
            s_E_int = t_E_int - 1
        else:
            raise ValueError("t_X and t_E must be both None or both not None.")

        t_X_float = t_X_int / self.T
        s_X_float = s_X_int / self.T
        t_E_float = t_E_int / self.T
        s_E_float = s_E_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t_X = self.noise_schedule(t_normalized=t_X_float)
        beta_t_E = self.noise_schedule(t_normalized=t_E_float)                         # (bs, 1)
        alpha_s_X_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_X_float)
        alpha_s_E_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_E_float)      # (bs, 1)
        alpha_t_X_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_X_float)
        alpha_t_E_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_E_float)      # (bs, 1)

        Qtb_X = self.transition_model.get_Qt_bar(alpha_t_X_bar, device=self.device)
        Qtb_E = self.transition_model.get_Qt_bar(alpha_t_E_bar, device=self.device)  # (bs, dx, dx_c, dx_c) (bs, de, de)
        assert (abs(Qtb_X.X.sum(dim=3) - 1.) < 1e-4).all(), Qtb_X.X.sum(dim=3) - 1
        assert (abs(Qtb_E.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = (X.permute(0, 2, 1, 3) @ Qtb_X.X).permute(0, 2, 1, 3)  # (bs, n, dx, dx_c)
        probE = E @ Qtb_E.E.unsqueeze(1)  # (bs, n, n, de)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xcdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_X_int': t_X_int, 't_X': t_X_float, 'beta_t_X': beta_t_X, 
                      'alpha_s_X_bar': alpha_s_X_bar, 'alpha_t_X_bar': alpha_t_X_bar,
                      't_E_int': t_E_int, 't_E': t_E_float, 'beta_t_E': beta_t_E, 
                      'alpha_s_E_bar': alpha_s_E_bar, 'alpha_t_E_bar': alpha_t_E_bar, 
                      'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data
    
    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (X, E, y)
           noisy_data: dict
           X, E, y, labels: (bs, n, dx, dx_c),  (bs, n, n, de), (bs, dy), (bs, n)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t_X = noisy_data['t_X']
        t_E = noisy_data['t_E']

        # 1. Compute node count distribution
        N = node_mask.sum(1).long() # node count (bs,)
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t_X, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll
    
    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device) # (bs, dx, dx_c, dx_c) (bs, de, de)

        # Compute transition probabilities
        probX = (X.permute(0, 2, 1, 3) @ Qtb.X).permute(0, 2, 1, 3)  # (bs, n, dx, dx_c)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de)
        assert probX.shape == X.shape

        bs, n, dx, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, dx, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)
    
    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb_X = self.transition_model.get_Qt_bar(noisy_data['alpha_t_X_bar'], self.device)
        Qsb_X = self.transition_model.get_Qt_bar(noisy_data['alpha_s_X_bar'], self.device) # s = t-1
        Qt_X = self.transition_model.get_Qt(noisy_data['beta_t_X'], self.device)

        Qtb_E = self.transition_model.get_Qt_bar(noisy_data['alpha_t_E_bar'], self.device)
        Qsb_E = self.transition_model.get_Qt_bar(noisy_data['alpha_s_E_bar'], self.device) # s = t-1
        Qt_E = self.transition_model.get_Qt(noisy_data['beta_t_E'], self.device)

        # Compute distributions to compare with KL
        bs, n, dx, _ = X.shape
        prob_true_X = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt_X, Qsb=Qsb_X, Qtb=Qtb_X)
        prob_true_E = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt_E, Qsb=Qsb_E, Qtb=Qtb_E)
        prob_true_E.E = prob_true_E.E.reshape((bs, n, n, -1))

        prob_pred_X = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt_X, Qsb=Qsb_X, Qtb=Qtb_X)
        prob_pred_E = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt_E, Qsb=Qsb_E, Qtb=Qtb_E)
        prob_pred_E.E = prob_pred_E.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        _, _, prob_pred_X.X, prob_pred_E.E = diffusion_utils.mask_distributions(true_X=prob_true_X.X,
                                                                                true_E=prob_true_E.E,
                                                                                pred_X=prob_pred_X.X,
                                                                                pred_E=prob_pred_E.E,
                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true_X.X, torch.log(prob_pred_X.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true_E.E, torch.log(prob_pred_E.E))
        return self.T * (kl_x + kl_e)
    
    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        # qx (bs, dx, dx_c, dx_c), qe (bs, de, de), qy (bs, dy, dy)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device) 

        probX0 = (X.permute(0, 2, 1, 3) @ Q0.X).permute(0, 2, 1, 3)  # (bs, n, dx, dx_c)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xcdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't_X': torch.zeros(X0.shape[0], 1).type_as(y0), 't_E': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xcdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)
    
    @torch.no_grad()
    def denoised_smoothing(self, dataloader, classifier=None, hparams=None):
        """
        hparams: dict
        """
        test_len = self.dataset_info.split_len['test']
        num_classes = len(self.dataset_info.node_types)
        batch_size = dataloader.batch_size
        device = classifier.device
        
        pre_votes_list = []
        votes_list = []
        targets_list = []
        
        for data in tqdm(dataloader, desc="Denoised Smoothing"):
            target = torch.tensor(data.label).to(device)
            pad = -torch.ones((batch_size - len(data.label)), dtype=torch.float).to(device)
            target = torch.cat((target, pad), dim=0)

            pre_votes = torch.zeros((batch_size, num_classes), dtype=torch.long, device=device)
            votes = torch.zeros((batch_size, num_classes), dtype=torch.long, device=device)

            pre_votes_list.append(self.denoise_pred(data, hparams['attr_noise_scale'], hparams['adj_noise_scale'], hparams['pre_n_samples'], pre_votes, classifier))
            votes_list.append(self.denoise_pred(data, hparams['attr_noise_scale'], hparams['adj_noise_scale'], hparams['n_samples'], votes, classifier))
            targets_list.append(target)        
        
        pre_votes = torch.cat(pre_votes_list, dim=0)[:test_len, :]
        votes = torch.cat(votes_list, dim=0)[:test_len, :]
        targets = torch.cat(targets_list, dim=0)[:test_len]
        pre_labels = pre_votes.argmax(-1)    
        labels = votes.argmax(-1)

        correct = (pre_labels == targets).cpu().numpy()
        clean_acc = correct.mean()
        print(f'Clean accuracy: {clean_acc}')
        majority_correct = (labels == targets).cpu().numpy()
        majority_acc = majority_correct.mean()
        print(f'Majority vote accuracy: {majority_acc}')
        
        certificate = {}
        if hparams['certify']:
            certificate = certify(majority_correct, pre_votes.cpu(), votes.cpu(), hparams)
        certificate['clean_acc'] = clean_acc
        certificate['majority_acc'] = majority_acc
        certificate['correct'] = correct.tolist()

        return certificate
    
    def denoise_pred(self, data, t_X, t_E, n_samples, votes, classifier):
        """
        Return: denoised prediction votes for data with classifier.
        """

        for _ in range(n_samples):
            denoised_data = self.denoise_Z(data, t_X, t_E)
            # print(f'denoised_data: {denoised_data[0].shape, denoised_data[1].shape, len(denoised_data[5])}')
            pred = classifier(denoised_data)

            row_indices = torch.arange(pred.size(0))
            votes[row_indices, pred.argmax(-1)] += 1
            # print(f'votes: {votes, pred.argmax(-1)}')
        return votes
        
    @torch.no_grad()
    def denoise_Z(self, data, t_X, t_E):
        """
        Receive data and denoise data of noise scale t.
        """
        data = data.to(self.device)
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask, t_X, t_E)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # pred = utils.PlaceHolder(X=torch.tensor(noisy_data['X_t'].clone().detach(), dtype=float), 
        #                          E=torch.tensor(noisy_data['E_t'].clone().detach(), dtype=float), 
        #                          y=torch.tensor(noisy_data['y_t'].clone().detach(), dtype=float))
        # pred = pred.mask(node_mask)
            
        unnormalized_prob_X = F.softmax(pred.X, dim=-1)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, dx, dx_c
        unnormalized_prob_E = F.softmax(pred.E, dim=-1)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)  # bs, n, dx, dx_c

        denoised_data = diffusion_utils.sample_discrete_features(probX=prob_X.cpu(), probE=prob_E.cpu(), node_mask=node_mask.cpu())
        x_list = []
        adj_list = []
        batch_size = denoised_data.X.size(0)
        for graph in range(batch_size):
            mask = node_mask[graph].cpu()
            x_list.append(denoised_data.X[graph][mask].float().cpu())
            adj_list.append(denoised_data.E[graph][mask][:, mask].float().cpu())

        data = general_utils.get_data(x_list, adj_list, data.label, node_mask)

        return data
    
    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=3).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()         # y is time conditioning + global feature
        return self.model(X, E, y, node_mask)
    
    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)

        extra_X = extra_features.X
        extra_E = extra_features.E
        extra_y = extra_features.y

        t_X = noisy_data['t_X']
        t_E = noisy_data['t_E']
        extra_y = torch.cat((extra_y, t_X, t_E), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
    
    def compute_noise(self, t_X, t_E):
        """ Compute noise for a given time step t. """

        t_X_int = torch.full((1, 1), t_X, device=self.device).float()
        t_X_float = t_X_int / self.T
        print(f'attribute noise scale: {t_X}/{self.T}')

        t_E_int = torch.full((1, 1), t_E, device=self.device).float()
        t_E_float = t_E_int / self.T
        print(f'adjacent noise scale: {t_E}/{self.T}')

        alpha_t_X_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_X_float)      # (1, 1)
        alpha_t_E_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_E_float)      # (1, 1)

        Qtb_X = self.transition_model.get_Qt_bar(alpha_t_X_bar, device=self.device)  # (1, dx, dx_c, dx_c) (1, de, de)
        Qtb_E = self.transition_model.get_Qt_bar(alpha_t_E_bar, device=self.device)  # (1, dx, dx_c, dx_c) (1, de, de)
        assert (abs(Qtb_X.X.sum(dim=3) - 1.) < 1e-4).all(), Qtb_X.X.sum(dim=3) - 1
        assert (abs(Qtb_E.E.sum(dim=2) - 1.) < 1e-4).all()

        X_flip_prob = Qtb_X.X.squeeze(0).mean(dim=0)
        E_flip_prob = Qtb_E.E.squeeze(0)
        print(f'X_p_plus: {X_flip_prob[0][1]:.4f}, X_p_minus: {X_flip_prob[1][0]:.4f}')
        print(f'E_p_plus: {E_flip_prob[0][1]:.4f}, E_p_minus: {E_flip_prob[1][0]:.4f}')
        return {'p': 1, 'smoothing_distribution': "sparse", 'append_indicator': False,
                'p_plus': X_flip_prob[0][1].item(), 'p_minus': X_flip_prob[1][0].item(),
                'p_plus_adj': E_flip_prob[0][1].item(), 'p_minus_adj': E_flip_prob[1][0].item()}