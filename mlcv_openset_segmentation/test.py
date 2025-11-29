import argparse
import csv
import logging
from pathlib import Path

import torch
import yaml
from torchvision.datasets import VOCSegmentation
import lightning as L

from mlcv_openset_segmentation.datasets.datamodule import StreetHazardsOEDataModule
from mlcv_openset_segmentation.models.model_base import BaseSemanticSegmentationModel
from mlcv_openset_segmentation.models.model_residual import ResidualPatternLearningModel
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
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="stats/test_metrics.csv",
        help="Path to CSV file where metrics will be appended.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Optional experiment name to store in metrics CSV.",
    )
    return parser.parse_args()


def load_backbone_model(checkpoint_path: Path):
    logger.info("Loading pretrained backbone from %s", checkpoint_path)
    backbone_module = BaseSemanticSegmentationModel.load_from_checkpoint(
        str(checkpoint_path)
    )
    backbone = backbone_module.model
    backbone.eval()
    return backbone


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

    if model_type == "residual":
        backbone_ckpt = Path(cfg["backbone"]["checkpoint_path"])
        backbone = load_backbone_model(backbone_ckpt)
        return ResidualPatternLearningModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            base_segmenter=backbone,
            outlier_class_idx=13,
            use_energy_entropy=cfg["model"]["use_energy_entropy"],
            score_type=cfg["model"]["score_type"],
            use_gaussian=cfg["model"]["use_gaussian_blur"],
            optimizer_params=cfg["optimizer"],
            scheduler_name=cfg["scheduler"]["name"],
            scheduler_params=cfg["scheduler"],
        )

    raise ValueError(f"Unknown model type '{model_type}'.")


def build_datamodule(config):
    _, eval_transform, _ = get_transforms(config["data"])
    oe_cfg = config["data"]["outlier_dataset"]

    voc_train = VOCSegmentation(
        root=oe_cfg["root"],
        year=oe_cfg["year"],
        image_set="train",
        download=oe_cfg["download"],
    )

    voc_val = VOCSegmentation(
        root=oe_cfg["root"],
        year=oe_cfg["year"],
        image_set="val",
        download=oe_cfg["download"],
    )

    return StreetHazardsOEDataModule(
        root_dir=config["data"]["root_dir"],
        outlier_train_dataset=voc_train,
        outlier_val_dataset=voc_val,
        train_transform=None,
        eval_transform=eval_transform,
    )


def append_metrics_to_csv(csv_path: Path, metrics: dict, experiment_name: str):
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    metrics_with_exp = {"experiment_name": experiment_name, **metrics}

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_with_exp.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_with_exp)

    logger.info("Metrics appended to %s", csv_path)


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
    test_results = trainer.test(model, datamodule=data_module)

    if isinstance(test_results, list) and test_results:
        metrics = test_results[0]
    else:
        metrics = {}

    if args.metrics_csv:
        append_metrics_to_csv(args.metrics_csv, metrics, args.experiment_name)


if __name__ == "__main__":
    main()
