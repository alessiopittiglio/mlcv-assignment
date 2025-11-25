import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchvision.datasets import VOCSegmentation

from mlcv_openset_segmentation.datamodule import StreetHazardsOEDataModule
from mlcv_openset_segmentation.dataset import StreetHazardsDataset
from mlcv_openset_segmentation.model_base import BaseSemanticSegmentationModel
from mlcv_openset_segmentation.model_residual import ResidualPatternLearningModel
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


def build_callbacks(save_dir: Path, early_cfg: dict):
    monitor_metric = early_cfg.get("metric", "val_loss")
    monitor_mode = early_cfg.get("mode", "min")

    model_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        save_top_k=1,
        mode=monitor_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [model_checkpoint, lr_monitor]

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

    rpl_cfg = cfg["rpl"]

    model = ResidualPatternLearningModel(
        base_segmenter=backbone,
        outlier_class_idx=StreetHazardsDataset.ANOMALY_ID,
        lr=rpl_cfg["lr"],
        use_energy_entropy=rpl_cfg["use_energy_entropy"],
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
    )

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
