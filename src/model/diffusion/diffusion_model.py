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

from src.model.diffusion.train_metrics import TrainLossDiscrete, NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.model.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.model.diffusion import utils
from src.model.diffusion import diffusion_utils


class GraphJointDiffuser(pl.LightningModule):
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
        self.print(f"Size of the input features:")
        self.print(f"node_attr={self.Xdim}, node_attr_classes={self.Xcdim}, edge_classes={self.Edim}, time_condition={self.ydim}, node_classes={self.labeldim}")
        self.print("Size of the output features")
        self.print(f"node_attr={self.Xdim_output}, node_attr_classes={self.Xcdim_output}, edge_classes={self.Edim_output}, time_condition={self.ydim_output}, node_classes={self.labeldim_output}")
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
    
    def training_step(self, data, i):
        return data
    
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