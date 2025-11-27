from copy import deepcopy

import lightning as L
import torch
from torchmetrics.classification import BinaryAveragePrecision
from torchvision.transforms import GaussianBlur

from mlcv_openset_segmentation.models.rpl import RPLDeepLab
from mlcv_openset_segmentation.models.rpl_losses import energy_entropy_loss, energy_loss


class ResidualPatternLearningModel(L.LightningModule):

    def __init__(
        self,
        base_segmenter: torch.nn.Module,
        outlier_class_idx: int,
        use_energy_entropy: bool = False,
        score_type: str = "energy_entropy",
        use_gaussian: bool = True,
        optimizer_params: dict = None,
        scheduler_name: str = None,
        scheduler_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_segmenter"])

        self.model = RPLDeepLab(base_segmenter)

        # Frozen backbone for vanilla logits
        self.backbone = deepcopy(base_segmenter).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.val_aupr = BinaryAveragePrecision(thresholds=200)
        self.test_aupr = BinaryAveragePrecision(thresholds=500)

        self.optimizer_params = optimizer_params or {}
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params or {}

        self.gaussian_blur = (
            GaussianBlur(kernel_size=7, sigma=1.0) if use_gaussian else None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        return self.model(x)

    def _compute_loss(
        self,
        rpl_logits: torch.Tensor,
        targets: torch.Tensor,
        vanilla_logits: torch.Tensor,
    ):
        loss_fn = (
            energy_entropy_loss if self.hparams.use_energy_entropy else energy_loss
        )

        return loss_fn(
            logits=rpl_logits,
            targets=targets,
            vanilla_logits=vanilla_logits,
            out_idx=self.hparams.outlier_class_idx,
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch

        with torch.no_grad():
            vanilla_logits = self.backbone(images)

        _, rpl_logits = self(images)
        loss = self._compute_loss(rpl_logits, targets, vanilla_logits)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        with torch.no_grad():
            vanilla_logits = self.backbone(images)

        _, rpl_logits = self(images)
        loss = self._compute_loss(rpl_logits, targets, vanilla_logits)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        anomaly_mask = (targets == self.hparams.outlier_class_idx).long()
        anomaly_scores = self._compute_anomaly_scores(rpl_logits)

        self.val_aupr.update(anomaly_scores.flatten(), anomaly_mask.flatten())
        self.log("val_aupr", self.val_aupr, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch

        with torch.no_grad():
            _, rpl_logits = self(images)

        anomaly_mask = (targets == self.hparams.outlier_class_idx).long()
        anomaly_scores = self._compute_anomaly_scores(rpl_logits)

        self.test_aupr.update(anomaly_scores.flatten(), anomaly_mask.flatten())
        self.log("test_aupr", self.test_aupr, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def _compute_anomaly_scores(self, logits):
        energy = -torch.logsumexp(logits, dim=1)

        if self.hparams.score_type == "energy":
            score = energy

        elif self.hparams.score_type == "energy_entropy":
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            score = energy + entropy

        else:
            raise ValueError(
                f"Invalid score_type: {self.hparams.score_type}. "
                "Expected 'energy' or 'energy_entropy'."
            )

        if self.gaussian_blur is not None:
            score = self.gaussian_blur(score.unsqueeze(0)).squeeze(0)

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.optimizer_params,
        )

        if self.scheduler_name is None:
            return optimizer

        if self.scheduler_name == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                **self.scheduler_params,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
