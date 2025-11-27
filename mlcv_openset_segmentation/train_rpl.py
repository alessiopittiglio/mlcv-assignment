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

from mlcv_openset_segmentation.datasets.datamodule import StreetHazardsOEDataModule
from mlcv_openset_segmentation.datasets.dataset import StreetHazardsDataset
from mlcv_openset_segmentation.models.model_base import BaseSemanticSegmentationModel
from mlcv_openset_segmentation.models.model_residual import ResidualPatternLearningModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RPL model for StreetHazards with Outlier Exposure."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def setup_scheduler(cfg, steps_per_epoch):
    max_epochs = cfg["trainer"]["max_epochs"]
    accumulate_batches = cfg["trainer"].get("accumulate_grad_batches", 1)

    optimizer_steps_per_epoch = steps_per_epoch // accumulate_batches
    total_steps = optimizer_steps_per_epoch * max_epochs

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_cfg["total_steps"] = total_steps
    cfg["scheduler"] = scheduler_cfg


def build_callbacks(save_dir: Path, early_cfg: dict):
    monitor_metric = early_cfg.get("monitor", "val_loss")
    monitor_mode = early_cfg.get("mode", "min")

    model_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        save_top_k=1,
        mode=monitor_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(**early_cfg)

    callbacks = [model_checkpoint, lr_monitor, early_stopping]

    return callbacks


def build_outlier_datasets(cfg):
    oe_cfg = cfg["data"]["outlier_dataset"]

    if oe_cfg["type"] != "voc":
        raise ValueError(f"Unsupported outlier dataset type: {oe_cfg['type']}")

    common_args = dict(
        root=oe_cfg["root"],
        year=oe_cfg["year"],
        download=oe_cfg["download"],
    )

    train_dataset = VOCSegmentation(
        image_set=oe_cfg["image_set"],
        **common_args,
    )
    val_dataset = VOCSegmentation(
        image_set="val",
        **common_args,
    )

    return train_dataset, val_dataset


def load_backbone_model(checkpoint_path: Path):
    logger.info("Loading pretrained backbone from %s", checkpoint_path)

    backbone_module = BaseSemanticSegmentationModel.load_from_checkpoint(
        str(checkpoint_path)
    )
    backbone = backbone_module.model
    backbone.eval()

    return backbone


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path)

    L.seed_everything(cfg["seed"])

    run = wandb.init(project="mlcv-assignment", config=cfg)
    wandb_logger = WandbLogger(experiment=run)

    train_transform, eval_transform, normalize_only = get_transforms(cfg["data"])

    backbone_ckpt = Path(cfg["backbone"]["checkpoint_path"])
    backbone = load_backbone_model(backbone_ckpt)

    model = ResidualPatternLearningModel(
        base_segmenter=backbone,
        outlier_class_idx=StreetHazardsDataset.ANOMALY_ID,
        use_energy_entropy=cfg["model"]["use_energy_entropy"],
        score_type=cfg["model"]["score_type"],
        use_gaussian=cfg["model"]["use_gaussian_blur"],
        optimizer_params=cfg["optimizer"],
        scheduler_name=cfg["scheduler"]["name"],
        scheduler_params=cfg["scheduler"],
    )

    voc_train, voc_val = build_outlier_datasets(cfg)

    data_cfg = cfg["data"]
    datamodule = StreetHazardsOEDataModule(
        root_dir=data_cfg["root_dir"],
        outlier_train_dataset=voc_train,
        outlier_val_dataset=voc_val,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_transform=train_transform,
        eval_transform=eval_transform,
        anomaly_normalize=normalize_only,
        inject_probability=data_cfg["anomaly_injection"]["probability"],
        max_anomalies_per_image=data_cfg["anomaly_injection"][
            "max_anomalies_per_image"
        ],
    )
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    steps_per_epoch = len(train_dataloader)
    setup_scheduler(cfg, steps_per_epoch)

    save_dir = Path(cfg["save_dir"]) / run.id
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(
        save_dir=save_dir, early_cfg=cfg.get("early_stopping", {})
    )

    csv_logger = CSVLogger(save_dir="logs", name=run.id)

    trainer = L.Trainer(
        logger=[wandb_logger, csv_logger],
        callbacks=callbacks,
        **cfg["trainer"],
    )

    logger.info("Starting RPL training...")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()
