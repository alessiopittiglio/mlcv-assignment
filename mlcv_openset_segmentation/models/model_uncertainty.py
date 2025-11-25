from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from mlcv_openset_segmentation.models.model_base import (
    BaseSemanticSegmentationModel,
    IGNORE_INDEX,
    ANOMALY_ID,
)
from mlcv_openset_segmentation.postprocessing import BoundarySuppressionWithSmoothing


class UncertaintyModel(BaseSemanticSegmentationModel):
    def __init__(
        self,
        num_classes: int = 13,
        encoder_name: str = "resnet50",
        optimizer_params: dict = None,
        scheduler_params: dict = None,
        uncertainty_type: str = "msp",
        sml_stats_path: str = "stats/sml_stats.pt",
        use_boundary_postprocessing: bool = False,
        boundary_postprocessing_params: dict = None,
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

        self.boundary_postprocessor = None
        if use_boundary_postprocessing:
            params = boundary_postprocessing_params or {}
            self.boundary_postprocessor = BoundarySuppressionWithSmoothing(**params)

        if self.hparams.uncertainty_type == "sml":
            self._load_sml_statistics(self.hparams.sml_stats_path)

    def _load_sml_statistics(self, stats_path: str) -> None:
        stats_file = Path(stats_path)
        if not stats_file.exists():
            raise FileNotFoundError(f"SML stats file not found: {stats_file}")

        stats = torch.load(stats_file, map_location="cpu")

        means = stats["means"].float().view(-1)
        stds = stats["stds"].float().view(-1).clamp_min(1e-6)

        self.register_buffer("sml_means", means, persistent=False)
        self.register_buffer("sml_stds", stds, persistent=False)

    def _apply_boundary_postprocessing(
        self,
        raw_scores: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        if self.boundary_postprocessor is None:
            return raw_scores

        processed = self.boundary_postprocessor(
            raw_scores.unsqueeze(1),
            prediction=predictions,
        )
        return processed

    @torch.no_grad()
    def _compute_anomaly_scores(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        utype = self.hparams.uncertainty_type

        if utype == "msp":
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            raw = self._apply_boundary_postprocessing(max_probs, predictions)
            return 1.0 - raw

        if utype == "max_logit":
            max_logits, _ = logits.max(dim=1)
            raw = self._apply_boundary_postprocessing(max_logits, predictions)
            return -raw

        if utype == "entropy":
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            return self._apply_boundary_postprocessing(entropy, predictions)

        if utype == "sml":
            max_logits, preds = logits.max(dim=1)
            means = self.sml_means[preds]
            stds = self.sml_stds[preds]
            sml_scores = (max_logits - means) / stds
            raw = self._apply_boundary_postprocessing(sml_scores, predictions)
            return -raw

        raise NotImplementedError(f"Unknown uncertainty type: '{utype}'")

    def test_step(self, batch, batch_idx):
        images, targets = batch

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        masked_targets = targets.clone()
        masked_targets[targets == ANOMALY_ID] = IGNORE_INDEX

        self.test_iou_closed.update(preds, masked_targets)
        self.log(
            "val_miou_closed",
            self.test_iou_closed,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        anomaly_mask = targets == ANOMALY_ID
        anomaly_scores = self._compute_anomaly_scores(logits, preds)

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
        self.log("val_aupr_anomaly", aupr, prog_bar=True, logger=True)

        self.pixel_anomaly_scores.clear()
        self.pixel_anomaly_labels.clear()
