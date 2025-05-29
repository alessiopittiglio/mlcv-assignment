import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .model_base import BaseSemanticSegmentationModel

ANOMALY_ID = 12

class UncertaintyModel(BaseSemanticSegmentationModel):
    def __init__(
            self,
            num_classes: int = 12,
            model_name: str = 'deeplabv3_resnet50',
            use_aux_loss: bool = True,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
            uncertainty_type='msp',
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            use_aux_loss=use_aux_loss,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.save_hyperparameters()

        self.all_pixel_anomaly_scores = []
        self.all_pixel_anomaly_labels = []

    def _calculate_anomaly_scores(self, logits_known: torch.Tensor) -> torch.Tensor:
        if self.hparams.uncertainty_type == 'msp':
            softmax_probs = torch.softmax(logits_known, dim=1)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            anomaly_scores = 1.0 - max_probs
        elif self.hparams.uncertainty_type == 'max_logit':
            max_logits, _ = torch.max(logits_known, dim=1)
            anomaly_scores = -max_logits
        elif self.hparams.uncertainty_type == 'entropy':
            softmax_probs = torch.softmax(logits_known, dim=1)
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-9), dim=1)
            anomaly_scores = entropy
        else:
            raise NotImplementedError(
                f"Method {self.hparams.uncertainty_type} not implemented"
            )
        return anomaly_scores

    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)

        images, gt_full_masks = batch
        outputs = self(images)
        logits_known = outputs['out']

        gt_anomaly_masks = (gt_full_masks == ANOMALY_ID).long()
        anomaly_scores = self._calculate_anomaly_scores(logits_known)

        pixel_anomaly_scores = anomaly_scores.detach().cpu().numpy().ravel()
        pixel_anomaly_labels = gt_anomaly_masks.detach().cpu().numpy().ravel()
        
        self.all_pixel_anomaly_scores.append(pixel_anomaly_scores)
        self.all_pixel_anomaly_labels.append(pixel_anomaly_labels)

    def on_test_epoch_end(self):
        all_scores = np.concatenate(self.all_pixel_anomaly_scores)
        all_labels = np.concatenate(self.all_pixel_anomaly_labels)

        aupr = average_precision_score(all_labels, all_scores)

        self.log('test_aupr_anomaly', aupr, prog_bar=True, logger=True)
