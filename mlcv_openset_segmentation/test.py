import argparse
import logging
from pathlib import Path

import torch
import yaml

import lightning as L

from mlcv_openset_segmentation.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.model_metric import MetricLearningModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a model trained for street hazard detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    return parser.parse_args()


def build_model_from_config(cfg, checkpoint_path):
    model_type = cfg["model"].get("type", "uncertainty").lower()

    if model_type == "uncertainty":
        model = UncertaintyModel.load_from_checkpoint(
            checkpoint_path,
            num_classes=cfg["data"]["num_classes"],
            model_name=cfg["model"]["model_name"],
            use_aux_loss=cfg["model"]["use_aux_loss"],
            uncertainty_type=cfg["model"]["uncertainty_type"],
            optimizer_kwargs=cfg["optimizer"],
            scheduler_kwargs=cfg["scheduler"],
            sml_stats_path="artifacts/sml_stats.pt",
            strict=False,
        )

    elif model_type == "metric":
        model = MetricLearningModel.load_from_checkpoint(
            checkpoint_path,
            num_classes=cfg["data"]["num_classes"],
            model_name=cfg["model"]["model_name"],
            use_aux_loss=cfg["model"]["use_aux_loss"],
            optimizer_kwargs=cfg["optimizer"],
            scheduler_kwargs=cfg["scheduler"],
            strict=False,
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

    L.seed_everything(cfg["seed"], workers=True)

    _, eval_transform = get_transforms(cfg["data"])

    data_module = StreetHazardsDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        eval_transform=eval_transform,
    )
    data_module.setup(stage="test")

    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = build_model_from_config(cfg, args.checkpoint_path)

    trainer = L.Trainer(**cfg["trainer"])

    logger.info("Starting test phase...")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
