import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import yaml

from mlcv_openset_segmentation.datasets.datamodule import StreetHazardsOEDataModule
from mlcv_openset_segmentation.models.model_metric import MetricLearningModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Metric Learning model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    L.seed_everything(cfg["seed"], workers=True)

    _, eval_transform, _ = get_transforms(cfg["data"])

    datamodule = StreetHazardsOEDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["test_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        eval_transform=eval_transform,
        outlier_train_dataset=None,
        outlier_val_dataset=None,
    )
    datamodule.setup(stage="test")

    model = MetricLearningModel(
        num_classes=cfg["data"]["num_classes"],
        encoder_path=cfg["model"]["encoder_path"],
        decoder_path=cfg["model"]["decoder_path"],
        in_channels=cfg["model"]["in_channels"],
    )

    trainer = L.Trainer(**cfg["trainer"])
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
