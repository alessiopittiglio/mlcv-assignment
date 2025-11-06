import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim

from .model_base import BaseSemanticSegmentationModel, IGNORE_INDEX, ANOMALY_ID


class DMLNetLoss(nn.Module):
    """
    Implements LDCE and LVL losses from Cen et al. (Deep Metric Learning for Open World
    Semantic Segmentation).
    """

    def __init__(self, lambda_vl: float = 0.01, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.lambda_vl = lambda_vl
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(
        self, embeddings: torch.Tensor, prototypes: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, emb_dim, h, w = embeddings.shape
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(n * h * w, emb_dim)
        masks_flat = masks.reshape(n * h * w)

        valid_mask = masks_flat != self.ignore_index
        embeddings_valid = embeddings_flat[valid_mask]
        masks_valid = masks_flat[valid_mask]

        if embeddings_valid.numel() == 0:
            zero = torch.tensor(0.0, device=embeddings.device)
            return zero, zero

        # LDCE: distance between embeddings and prototypes
        dists_sq = torch.cdist(embeddings_valid, prototypes, p=2.0) ** 2
        logits_dce = -dists_sq
        loss_dce = self.ce_loss(logits_dce, masks_valid).mean()

        # LVL: intra-class variance regularization
        prototypes_correct = prototypes[masks_valid]
        loss_lvl = ((embeddings_valid - prototypes_correct) ** 2).sum(dim=1).mean()

        return loss_dce, loss_lvl


class MetricLearningModel(BaseSemanticSegmentationModel):
    def __init__(
        self,
        embedding_dim: int = 12,
        lambda_vl: float = 0.01,
        beta_mixture: float = 20.0,
        gamma_mixture: float = 0.8,
        num_classes: int = 13,
        model_name: str = "deeplabv3_resnet50",
        use_aux_loss: bool = True,
        optimizer_kwargs: dict = None,
        scheduler_kwargs: dict = None,
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            use_aux_loss=use_aux_loss,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.save_hyperparameters(
            "embedding_dim", "lambda_vl", "beta_mixture", "gamma_mixture"
        )

        if isinstance(self.model.classifier, nn.Sequential):
            aspp_channels = 256

            self.embedding_head = nn.Sequential(
                nn.Conv2d(
                    aspp_channels,
                    aspp_channels // 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(aspp_channels // 2),
                nn.ReLU(),
                nn.Conv2d(
                    aspp_channels // 2, self.hparams.embedding_dim, kernel_size=1
                ),
            )
            self.feature_source = "aspp"

            self.prototypes = nn.Parameter(
                torch.randn(self.hparams.num_classes, self.hparams.embedding_dim)
            )

            self.loss_fn = DMLNetLoss(
                lambda_vl=self.hparams.lambda_vl, ignore_index=IGNORE_INDEX
            )

            self.test_aupr_anomaly = torchmetrics.AveragePrecision(task="binary")

    def extract_aspp_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ASPP features before the final classifier."""
        features = self.model.backbone(x)
        return self.model.classifier[0](features["out"])

    def forward(self, x: torch.Tensor) -> dict:
        input_shape = x.shape[-2:]

        # Backbone + ASPP
        aspp_features = self.extract_aspp_features(x)

        # Main segmentation head
        seg_logits = self.model.classifier[-1](aspp_features)
        seg_logits = F.interpolate(
            seg_logits, size=input_shape, mode="bilinear", align_corners=False
        )

        outputs = {"out": seg_logits}

        # Embedding head
        embeddings = self.embedding_head(aspp_features)
        embeddings = F.interpolate(
            embeddings,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )
        outputs["embeddings"] = embeddings

        # Auxiliary head (if present)
        if (
            self.hparams.use_aux_loss
            and hasattr(self.model, "aux_classifier")
            and self.model.aux_classifier is not None
        ):
            features = self.model.backbone(x)
            if "aux" in features:
                aux_logits = self.model.aux_classifier(features["aux"])
                aux_logits = F.interpolate(
                    aux_logits, size=input_shape, mode="bilinear", align_corners=False
                )
                outputs["aux"] = aux_logits

        return outputs

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, gt_masks = batch
        outputs = self(images)

        embeddings = outputs["embeddings"]
        loss_dce, loss_lvl = self.loss_fn(embeddings, self.prototypes, gt_masks)
        total_loss = loss_dce + self.hparams.lambda_vl * loss_lvl

        self.log_dict(
            {
                "train_loss_dce": loss_dce,
                "train_loss_lvl": loss_lvl,
                "train_total_dml_loss": total_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Metric learning prediction
        with torch.no_grad():
            n, d, h, w = embeddings.shape
            emb_flat = embeddings.permute(0, 2, 3, 1).reshape(n * h * w, d)
            dists_sq = torch.cdist(emb_flat, self.prototypes, p=2.0) ** 2
            preds = torch.argmin(dists_sq, dim=1).reshape(n, h, w)

        self.train_miou.update(preds, gt_masks)
        self.log(
            "train_miou", self.train_miou, on_epoch=True, prog_bar=True, logger=True
        )
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int):
        images, gt_masks = batch
        outputs = self(images)

        embeddings = outputs["embeddings"]
        loss_dce, loss_lvl = self.loss_fn(embeddings, self.prototypes, gt_masks)
        total_loss = loss_dce + self.hparams.lambda_vl * loss_lvl

        self.log_dict(
            {
                "val_loss_dce": loss_dce,
                "val_loss_lvl": loss_lvl,
                "val_total_dml_loss": total_loss,
            },
            prog_bar=True,
            logger=True,
        )

        n, d, h, w = embeddings.shape
        emb_flat = embeddings.permute(0, 2, 3, 1).reshape(n * h * w, d)
        dists_sq = torch.cdist(emb_flat, self.prototypes, p=2.0) ** 2
        preds = torch.argmin(dists_sq, dim=1).reshape(n, h, w)

        self.val_miou.update(preds, gt_masks)
        self.log("val_miou", self.val_miou, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: tuple, batch_idx: int):
        images, gt_masks_full = batch
        outputs = self(images)

        seg_logits = outputs["out"]
        embeddings = outputs["embeddings"]

        # Closed-set mIoU
        gt_closed = gt_masks_full.clone()
        gt_closed[gt_masks_full == ANOMALY_ID] = IGNORE_INDEX
        preds_closed = seg_logits.argmax(dim=1)
        self.test_miou_closed.update(preds_closed, gt_closed)
        self.log("test_miou_closed", self.test_miou_closed, on_epoch=True, logger=True)

        # Compute anomaly scores (EDS + MMSP + mixture)
        n, d, h, w = embeddings.shape
        emb_flat = embeddings.permute(0, 2, 3, 1).reshape(n * h * w, d)

        dists_sq = torch.cdist(emb_flat, self.prototypes.detach(), p=2.0) ** 2
        prob_metric = F.softmax(-dists_sq, dim=1)
        max_prob_metric, _ = prob_metric.max(dim=1)
        p_mmsp = 1.0 - max_prob_metric

        s_xij = dists_sq.sum(dim=1).reshape(n, h * w)
        max_sx = s_xij.max(dim=1, keepdim=True)[0].clamp_min(1e-6)
        p_eds = 1.0 - (s_xij / max_sx).flatten()

        alpha = torch.sigmoid(
            self.hparams.beta_mixture * (p_eds - self.hparams.gamma_mixture)
        )
        final_scores = alpha * p_eds + (1 - alpha) * p_mmsp

        target_anomaly = (gt_masks_full == ANOMALY_ID).long().flatten()
        self.test_aupr_anomaly.update(final_scores, target_anomaly)
        self.log(
            "test_aupr_anomaly", self.test_aupr_anomaly, on_epoch=True, logger=True
        )

    def configure_optimizers(self):
        params = [
            {"params": self.model.backbone.parameters()},
            {"params": self.model.classifier[0].parameters()},
            {"params": self.embedding_head.parameters()},
            {"params": [self.prototypes]},
        ]

        optimizer_cfg = self.hparams.optimizer_kwargs or {}
        optimizer = optim.AdamW(params, **optimizer_cfg)

        scheduler_cfg = self.hparams.scheduler_kwargs or {}
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_dml_loss",
                "frequency": 1,
            },
        }
