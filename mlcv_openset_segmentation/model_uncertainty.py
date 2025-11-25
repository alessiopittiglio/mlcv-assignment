from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from mlcv_openset_segmentation.model_base import (
    BaseSemanticSegmentationModel,
    IGNORE_INDEX,
    ANOMALY_ID,
)


class UncertaintyModel(BaseSemanticSegmentationModel):
    def __init__(
        self,
        num_classes: int = 13,
        encoder_name: str = "resnet50",
        optimizer_params: dict = None,
        scheduler_params: dict = None,
        uncertainty_type: str = "msp",
        sml_stats_path: str = "stats/sml_stats.pt",
    ):
        super().__init__(
            num_classes=num_classes,
            encoder_name=encoder_name,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
        )
        self.save_hyperparameters()

        self.pixel_anomaly_scores = []
        self.pixel_anomaly_labels = []

        if self.hparams.uncertainty_type == "sml":
            stats_path = Path(sml_stats_path)
            if not stats_path.exists():
                raise FileNotFoundError(f"SML stats file not found at {stats_path}")
            stats = torch.load(sml_stats_path, map_location="cpu")

            means = stats["means"].float().view(-1)
            stds = stats["stds"].float().view(-1).clamp_min(1e-6)

            self.register_buffer("sml_means", means, persistent=False)
            self.register_buffer("sml_stds", stds, persistent=False)

    @torch.no_grad()
    def _compute_anomaly_scores(self, logits: torch.Tensor) -> torch.Tensor:
        utype = self.hparams.uncertainty_type

        if utype == "msp":
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            return 1.0 - max_probs

        if utype == "max_logit":
            max_logits, _ = logits.max(dim=1)
            return -max_logits

        if utype == "entropy":
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            return entropy

        if utype == "sml":
            max_logits, preds = logits.max(dim=1)
            means = self.sml_means[preds]
            stds = self.sml_stds[preds]
            sml_scores = (max_logits - means) / stds
            return -sml_scores

        raise NotImplementedError(f"Unknown uncertainty type: '{utype}'")

    def test_step(self, batch, batch_idx):
        images, targets = batch

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        masked_targets = targets.clone()
        masked_targets[targets == ANOMALY_ID] = IGNORE_INDEX

        self.test_iou_closed.update(preds, masked_targets)
        self.log(
            "test_miou_closed",
            self.test_iou_closed,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        anomaly_mask = targets == ANOMALY_ID
        anomaly_scores = self._compute_anomaly_scores(logits)

        self.pixel_anomaly_scores.append(anomaly_scores.cpu().float().numpy().ravel())
        self.pixel_anomaly_labels.append(
            anomaly_mask.cpu().numpy().astype(np.uint8).ravel()
        )

    def on_test_epoch_end(self):
        if not self.pixel_anomaly_scores:
            return
        all_scores = np.concatenate(self.pixel_anomaly_scores)
        all_labels = np.concatenate(self.pixel_anomaly_labels)

        aupr = average_precision_score(all_labels, all_scores)
        self.log("test_aupr_anomaly", aupr, prog_bar=True, logger=True)

        self.pixel_anomaly_scores.clear()
        self.pixel_anomaly_labels.clear()
