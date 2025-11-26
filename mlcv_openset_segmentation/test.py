import argparse
import logging
from pathlib import Path

import torch
import yaml
from torchvision.datasets import VOCSegmentation

import lightning as L

from mlcv_openset_segmentation.datasets.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.models.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.transforms import get_transforms

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a model trained for street hazard detection."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
    return parser.parse_args()


def build_model(cfg, checkpoint_path):
    model_type = cfg["model"].get("type", "uncertainty").lower()

    if model_type == "uncertainty":
        return UncertaintyModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            num_classes=cfg["data"]["num_classes"],
            encoder_name=cfg["model"]["encoder_name"],
            loss_type=cfg["model"]["loss_type"],
            uncertainty_type=cfg["model"]["uncertainty_type"],
            optimizer_params=cfg["optimizer"],
            scheduler_params=cfg["scheduler"],
            use_boundary_postprocessing=cfg["model"]["use_boundary_postprocessing"],
            boundary_postprocessing_params=cfg["model"][
                "boundary_postprocessing_params"
            ],
            strict=False,
        )

    raise ValueError(f"Unknown model type '{model_type}'.")


def build_datamodule(config):
    _, eval_tf, normalize_only = get_transforms(config["data"])

    oe_cfg = config["data"]["outlier_dataset"]
    voc_val = VOCSegmentation(
        root=oe_cfg["root"],
        year=oe_cfg["year"],
        image_set="val",
        download=oe_cfg["download"],
    )

    return StreetHazardsDataModule(
        root_dir=config["data"]["root_dir"],
        batch_size=config["data"]["test_batch_size"],
        num_workers=config["data"]["num_workers"],
        eval_transform=eval_tf,
        outlier_val_dataset=voc_val,
        anomaly_normalize=normalize_only,
    )


def main():
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    L.seed_everything(cfg["seed"], workers=True)

    data_module = build_datamodule(cfg)
    data_module.setup(stage="test")

    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = build_model(cfg, args.checkpoint_path)

    trainer = L.Trainer(**cfg["trainer"])

    logger.info("Starting test phase...")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
