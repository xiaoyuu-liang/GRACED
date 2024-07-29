import torch
torch.cuda.empty_cache()
import numpy as np
import argparse
import warnings
import logging
import hydra
import os
import pathlib

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)

from src.model.diffusion import utils
from src.model.diffusion.attributed_dataset import AttributedGraphDataModule, AttributedDatasetInfos
from src.model.diffusion.extra_features import DummyExtraFeatures
from src.model.diffusion.diffusion_model import GraphJointDiffuser
from src.model.diffusion.train_metrics import TrainAbstractMetricsDiscrete


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    resume_path = os.path.join(root_dir, cfg.general.test_only)

    model = GraphJointDiffuser.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        if category == 'denoiser':
            continue
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '/test_only'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = GraphJointDiffuser.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        if category == 'denoiser':
            continue
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '/test_only'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    denoiser_config = cfg["denoiser"]

    datamodule = AttributedGraphDataModule(cfg)
    dataset_infos = AttributedDatasetInfos(datamodule, dataset_config)
    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    
    train_metrics = TrainAbstractMetricsDiscrete()

    extra_features = DummyExtraFeatures()
    model_kwargs = {'dataset_infos': dataset_infos, 'extra_features': extra_features,
                    'train_metrics': train_metrics}
    
    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
    
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{dataset_config['name']}/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{dataset_config['name']}/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    use_gpu = (isinstance(cfg.general.gpus, str) or cfg.general.gpus > 0) and torch.cuda.is_available()
    device = 'cuda:' + cfg.general.gpus if isinstance(cfg.general.gpus, str) else 'cuda:0' if cfg.general.gpus > 0 else 'cpu'
    print(f"Using device {device}")

    model = GraphJointDiffuser(cfg, **model_kwargs)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="auto",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])
    
    if not cfg.general.test_only: # train / finetune
        print(f"Training {name}")
        trainer.fit(model, 
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(), 
                    ckpt_path=cfg.general.resume)


if __name__ == '__main__':
    main()