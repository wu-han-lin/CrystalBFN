from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    
)
from pytorch_lightning.loggers import WandbLogger
# from cdvae.common.ema import EMA
from crysbfn.common.callbacks import GenEvalCallback, Gradient_clip, EMACallback, CSPEvalCallback
# from cdvae.common.ema import EMA
from crysbfn.common.utils import log_hyperparameters, PROJECT_ROOT
import warnings
import multiprocess as mp
mp.set_start_method('spawn', force=True)

warnings.filterwarnings("ignore", category=UserWarning)
hydra.utils.log.info("ignore UserWarnings")

def build_callbacks(cfg: DictConfig, datamodule=None) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                # monitor='metric_amsd_precision',
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )
    # callbacks.append(ReconEvalCallback(), )
    
    callbacks.append(GenEvalCallback(cfg=cfg, datamodule=datamodule), )
    callbacks.append(CSPEvalCallback(cfg=cfg, datamodule=datamodule), )
    # if cfg.train.use_queue_clip:
    hydra.utils.log.info("Adding callback <GradientClip>")
    callbacks.append(Gradient_clip(use_queue_clip=cfg.train.use_queue_clip), )
    
    if "enable" in cfg.train.ema and cfg.train.ema.enable:
        hydra.utils.log.info("Adding callback <EMACallback>")
        callbacks.append(EMACallback(decay=cfg.train.ema.decay,ema_device='cuda'),)
    # callbacks.append(EMA(decay=0.9999,))
    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    
    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    model.cart_scaler = datamodule.cart_scaler.copy()
    model.train_loader = datamodule.train_dataloader()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    torch.save(datamodule.cart_scaler, hydra_dir / 'cart_coord_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg,datamodule=datamodule)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    if cfg.train.resume == True:
        # Load checkpoint (if exist)
        ckpts = list(hydra_dir.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[-2].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            hydra.utils.log.info(f"found checkpoint: {ckpt}")
            # 找到当时的hparams.yaml并load
            hparam_path = list(hydra_dir.glob('*.yaml'))[0]
            cfg = OmegaConf.load(hparam_path)
            hydra.utils.log.info(f"load hparams.yaml: {hparam_path}")
        else:
            raise FileNotFoundError(f"no checkpoint found in {ckpts}")
            ckpt = None
    else:
        ckpt = None
          
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        # val_check_interval = cfg.logging.val_check_interval_step,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        gradient_clip_val=cfg.train.gradient_clip_val if cfg.train.use_queue_clip else 0.,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
