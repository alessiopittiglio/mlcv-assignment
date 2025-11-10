import argparse
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from mlcv_openset_segmentation.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.transforms import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save mean and std for Standardized Max Logits (SML)."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the YAML configuration file used for training.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/sml_stats.pt",
        help="Path to save the computed statistics file.",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_statistics(model, dataloader, num_classes, device):
    model.eval()
    model.to(device)

    logger.info(
        f"Starting statistics computation on {len(dataloader.dataset)} samples..."
    )

    counts = torch.zeros(num_classes, dtype=torch.double, device=device)
    means = torch.zeros(num_classes, dtype=torch.double, device=device)
    m2s = torch.zeros(num_classes, dtype=torch.double, device=device)

    for images, _ in tqdm(dataloader, desc="Computing SML statistics"):
        images = images.to(device)
        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

        max_logits, _ = torch.max(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        for cls in range(num_classes):
            cls_logits = max_logits[preds == cls]
            n = cls_logits.numel()
            if n == 0:
                continue

            cls_mean = cls_logits.mean().double()
            cls_var = cls_logits.var(unbiased=False).double()

            # Welford's online algorithm
            delta = cls_mean - means[cls]
            total_n = counts[cls] + n

            means[cls] += delta * n / total_n
            m2s[cls] += cls_var * n + delta**2 * counts[cls] * n / total_n
            counts[cls] = total_n

        del images, logits, max_logits, preds  # cleanup for memory efficiency

    variances = m2s / counts.clamp_min(1)
    variances.clamp_(min=0.0)
    stds = torch.sqrt(variances)
    means = means.float()
    stds = stds.float().clamp_min(1e-6)

    return {"means": means, "stds": stds}


def main():
    args = parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_transform, _ = get_transforms(cfg["data"])
    data_module = StreetHazardsDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_transform,
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()

    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = UncertaintyModel.load_from_checkpoint(args.checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    num_classes = cfg["data"]["num_classes"]
    stats = compute_statistics(model, train_loader, num_classes, device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)

    logger.info(f"Statistics saved successfully to {output_path}")


if __name__ == "__main__":
    main()
