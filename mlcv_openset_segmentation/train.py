import argparse
import logging
from pathlib import Path

import torch
import wandb
import yaml

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from mlcv_openset_segmentation.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.model_metric import MetricLearningModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for street hazard detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


def build_model(cfg):
    model_type = cfg["model"].get("type", "uncertainty").lower()

    if model_type == "uncertainty":
        model = UncertaintyModel(
            num_classes=cfg["data"]["num_classes"],
            model_name=cfg["model"]["model_name"],
            use_aux_loss=cfg["model"]["use_aux_loss"],
            uncertainty_type=cfg["model"]["uncertainty_type"],
            optimizer_kwargs=cfg["optimizer"],
            scheduler_kwargs=cfg["scheduler"],
        )

    elif model_type == "metric":
        model = MetricLearningModel(
            num_classes=cfg["data"]["num_classes"],
            model_name=cfg["model"]["model_name"],
            use_aux_loss=cfg["model"]["use_aux_loss"],
            optimizer_kwargs=cfg["optimizer"],
            scheduler_kwargs=cfg["scheduler"],
        )

    else:
        raise ValueError(f"Unknown model type '{model_type}'.")

    return model


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run = wandb.init(project="mlcv-assignment")
    run_id = run.id
    logger = WandbLogger(experiment=run, config=cfg)

    L.seed_everything(cfg["seed"], workers=True)

    train_transform, eval_transform = get_transforms(cfg["data"])

    data_module = StreetHazardsDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["train_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    model = build_model(cfg)

    save_dir = Path("checkpoints/")
    save_dir /= run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    monitor_metric = early_cfg.get("metric", "val_loss")
    monitor_patience = early_cfg.get("patience", 5)
    monitor_mode = early_cfg.get("mode", "min")

    model_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        save_top_k=1,
        monitor=monitor_metric,
        mode=monitor_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [model_checkpoint, lr_monitor]

    if early_enabled:
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=monitor_patience,
            mode=monitor_mode,
        )
        callbacks.append(early_stopping)

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg["trainer"],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()
