import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchvision.datasets import VOCSegmentation

from mlcv_openset_segmentation.datasets.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.models.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for street hazard detection."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def build_model(cfg):
    model_type = cfg["model"].get("type", "uncertainty").lower()

    if model_type == "uncertainty":
        return UncertaintyModel(
            num_classes=cfg["data"]["num_classes"],
            encoder_name=cfg["model"]["encoder_name"],
            loss_type=cfg["model"]["loss_type"],
            uncertainty_type=cfg["model"]["uncertainty_type"],
            optimizer_params=cfg["optimizer"],
            scheduler_params=cfg["scheduler"],
        )

    raise ValueError(f"Unknown model type '{model_type}'.")


def setup_scheduler(cfg, steps_per_epoch):
    max_epochs = cfg["trainer"]["max_epochs"]
    total_steps = steps_per_epoch * max_epochs

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_cfg.setdefault("total_steps", total_steps)
    cfg["scheduler"] = scheduler_cfg


def build_callbacks(save_dir, early_cfg):
    monitor_metric = early_cfg.get("metric", "val_loss")
    monitor_patience = early_cfg.get("patience", 5)
    monitor_mode = early_cfg.get("mode", "min")
    early_enabled = early_cfg.get("enabled", False)

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

    return callbacks


def build_datamodule(config):
    train_tf, eval_tf, normalize_only = get_transforms(config["data"])

    oe_cfg = config["data"]["outlier_dataset"]
    voc_val = VOCSegmentation(
        root=oe_cfg["root"],
        year=oe_cfg["year"],
        image_set="val",
        download=oe_cfg["download"],
    )

    return StreetHazardsDataModule(
        root_dir=config["data"]["root_dir"],
        batch_size=config["data"]["train_batch_size"],
        num_workers=config["data"]["num_workers"],
        train_transform=train_tf,
        eval_transform=eval_tf,
        outlier_val_dataset=voc_val,
        anomaly_normalize=normalize_only,
    )


def main():
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    L.seed_everything(cfg["seed"], workers=True)

    run = wandb.init(project="mlcv-assignment", config=cfg)
    wandb_logger = WandbLogger(experiment=run)

    data_module = build_datamodule(cfg)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    steps_per_epoch = len(train_dataloader)
    setup_scheduler(cfg, steps_per_epoch)

    model = build_model(cfg)

    save_dir = Path("checkpoints") / run.id
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(
        save_dir=save_dir,
        early_cfg=cfg.get("early_stopping", {}),
    )

    csv_logger = CSVLogger(save_dir="logs", name=run.id)

    trainer = L.Trainer(
        logger=[wandb_logger, csv_logger],
        callbacks=callbacks,
        **cfg["trainer"],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()
