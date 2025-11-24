import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from mlcv_openset_segmentation.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.model_uncertainty import UncertaintyModel, ANOMALY_ID
from mlcv_openset_segmentation.transforms import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistogramQuantiles:
    def __init__(self, num_classes, num_bins=200, vmin=-20.0, vmax=20.0):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.edges = torch.linspace(vmin, vmax, num_bins + 1)
        self.hist_id = torch.zeros(num_classes, num_bins)
        self.hist_ood = torch.zeros(num_classes, num_bins)

    def update(self, values, cls, is_ood):
        hist = torch.histc(
            values.cpu(), bins=self.num_bins, min=self.vmin, max=self.vmax
        )
        if is_ood:
            self.hist_ood[cls] += hist
        else:
            self.hist_id[cls] += hist

    def _compute_quantile(self, hist, q):
        cdf = torch.cumsum(hist, dim=0)
        total = cdf[-1].item()

        if total == 0:
            return None

        idx = torch.searchsorted(cdf, q * total)
        idx = min(idx.item(), self.num_bins - 1)
        return self.edges[idx].item()

    def _compute_mean(self, hist):
        total = hist.sum().item()

        if total == 0:
            return None

        centers = (self.edges[:-1] + self.edges[1:]) / 2
        return float((hist * centers).sum().item() / total)

    def _compute_fpr95(self, hist_id, hist_ood):
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        cdf_id = torch.cumsum(hist_id, dim=0)

        total_id = cdf_id[-1].item()
        if total_id == 0:
            return None

        idx = torch.searchsorted(cdf_id, 0.95 * total_id)
        idx = min(idx.item(), len(hist_id) - 1)
        threshold = centers[idx].item()

        total_ood = hist_ood.sum().item()
        if total_ood == 0:
            fpr = 0.0
        else:
            mask = centers >= threshold
            fpr = float(hist_ood[mask].sum().item() / total_ood)

        return fpr, threshold

    def compute_global_fpr95(self):
        hist_id_global = self.hist_id.sum(dim=0)
        hist_ood_global = self.hist_ood.sum(dim=0)

        return self._compute_fpr95(hist_id_global, hist_ood_global)

    def finalize(self):
        stats = {}

        for cls in range(self.num_classes):
            h_id = self.hist_id[cls]
            h_ood = self.hist_ood[cls]

            stats[cls] = {
                "ID_Q1": self._compute_quantile(h_id, 0.25),
                "ID_Q3": self._compute_quantile(h_id, 0.75),
                "OOD_Q1": self._compute_quantile(h_ood, 0.25),
                "OOD_Q3": self._compute_quantile(h_ood, 0.75),
                "ID_mean": self._compute_mean(h_id),
                "OOD_mean": self._compute_mean(h_ood),
            }

        global_fpr, global_thresh = self.compute_global_fpr95()
        stats["GLOBAL_FPR95"] = global_fpr
        stats["GLOBAL_THRESHOLD95"] = global_thresh

        return stats


@torch.no_grad()
def compute_all_statistics(model, dataloader, num_classes, means, stds, device):
    model.eval()
    model.to(device)

    q_msp = HistogramQuantiles(num_classes, vmin=0.0, vmax=1.0)
    q_logit = HistogramQuantiles(num_classes, vmin=-5.0, vmax=35.0)
    q_sml = HistogramQuantiles(num_classes, vmin=-12.0, vmax=25.0)

    for images, labels in tqdm(dataloader, desc="Computing full statistics"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)["out"]
        probs = F.softmax(logits, dim=1)

        max_prob, preds = probs.max(dim=1)
        max_logit, _ = logits.max(dim=1)
        sml = (max_logit - means[preds]) / stds[preds]

        is_ood = labels == ANOMALY_ID
        is_id = ~is_ood

        for cls in range(num_classes):
            mask_id = (preds == cls) & is_id
            mask_ood = (preds == cls) & is_ood

            if mask_id.any():
                q_msp.update(max_prob[mask_id], cls, is_ood=False)
                q_logit.update(max_logit[mask_id], cls, is_ood=False)
                q_sml.update(sml[mask_id], cls, is_ood=False)

            if mask_ood.any():
                q_msp.update(max_prob[mask_ood], cls, is_ood=True)
                q_logit.update(max_logit[mask_ood], cls, is_ood=True)
                q_sml.update(sml[mask_ood], cls, is_ood=True)

    return {
        "quantiles_msp": q_msp.finalize(),
        "quantiles_logit": q_logit.finalize(),
        "quantiles_sml": q_sml.finalize(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute class-wise uncertainty statistics"
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-path", default="stats/full_stats.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _, eval_transform = get_transforms(cfg["data"])

    data_module = StreetHazardsDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["test_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        eval_transform=eval_transform,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    model = UncertaintyModel.load_from_checkpoint(args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sml_stats_path = Path("stats/sml_stats.pt")
    stats = torch.load(sml_stats_path)

    means = stats["means"].to(device)
    stds = stats["stds"].to(device)

    results = compute_all_statistics(
        model=model,
        dataloader=test_loader,
        num_classes=cfg["data"]["num_classes"],
        means=means,
        stds=stds,
        device=device,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)

    logger.info(f"Full statistics saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
