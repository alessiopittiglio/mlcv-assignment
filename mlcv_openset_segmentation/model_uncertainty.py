from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .model_base import BaseSemanticSegmentationModel

ANOMALY_ID = 13


class UncertaintyModel(BaseSemanticSegmentationModel):
    def __init__(
        self,
        num_classes: int = 13,
        model_name: str = "deeplabv3_resnet50",
        use_aux_loss: bool = True,
        optimizer_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        uncertainty_type="msp",
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            use_aux_loss=use_aux_loss,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.save_hyperparameters()

        self.pixel_anomaly_scores = []
        self.pixel_anomaly_labels = []

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

        raise NotImplementedError(f"Unknown uncertainty type: '{utype}'")

    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)

        images, gt_masks = batch
        outputs = self(images)
        logits = outputs["out"]

        anomaly_mask = (gt_masks == ANOMALY_ID).long()
        anomaly_scores = self._compute_anomaly_scores(logits)

        self.pixel_anomaly_scores.append(anomaly_scores.detach().cpu().numpy().ravel())
        self.pixel_anomaly_labels.append(anomaly_mask.detach().cpu().numpy().ravel())

    def on_test_epoch_end(self):
        all_scores = np.concatenate(self.pixel_anomaly_scores)
        all_labels = np.concatenate(self.pixel_anomaly_labels)

        aupr = average_precision_score(all_labels, all_scores)
        self.log("test_aupr_anomaly", aupr, prog_bar=True, logger=True)
