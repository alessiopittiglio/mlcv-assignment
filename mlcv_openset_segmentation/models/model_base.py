import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, LovaszLoss

IGNORE_INDEX = 255
ANOMALY_ID = 13


class CombinedLoss(nn.Module):
    def __init__(
        self,
        loss_a: nn.Module,
        loss_b: nn.Module,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.weight_a * self.loss_a(
            logits, targets
        ) + self.weight_b * self.loss_b(logits, targets)


def build_loss(loss_type: str) -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    if loss_type == "dice":
        return DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

    if loss_type == "focal":
        return FocalLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

    if loss_type == "lovasz":
        return LovaszLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

    if loss_type == "ce_dice":
        ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        dice = DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
        return CombinedLoss(ce, dice)

    raise ValueError(
        f"Unsupported loss '{loss_type}'. "
        "Choose from: ce, dice, focal, lovasz, ce_dice"
    )


class BaseSemanticSegmentationModel(L.LightningModule):
    """
    Base class for semantic segmentation models using PyTorch Lightning.
    """

    def __init__(
        self,
        num_classes: int = 13,
        encoder_name: str = "resnet50",
        loss_type: str = "ce",
        optimizer_params: dict = None,
        scheduler_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
            activation=None,
        )

        self.loss_fn = build_loss(loss_type)

        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro",
            "ignore_index": IGNORE_INDEX,
        }

        self.train_iou = torchmetrics.JaccardIndex(**metric_args)
        self.val_iou = torchmetrics.JaccardIndex(**metric_args)
        self.test_iou_closed = torchmetrics.JaccardIndex(**metric_args)

        self.optimizer_params = optimizer_params or {"lr": 2e-4}
        self.scheduler_params = scheduler_params or {}

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch

        logits = self(images)
        loss = self.loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.train_iou.update(preds, targets)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.log(
            "train_miou",
            self.train_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        images, targets = batch

        logits = self(images)
        loss = self.loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.val_iou.update(preds, targets)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_miou", self.val_iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int):
        images, targets = batch

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        masked_targets = targets.clone()
        masked_targets[targets == ANOMALY_ID] = IGNORE_INDEX

        self.test_iou_closed.update(preds, masked_targets)

        self.log("test_miou_closed", self.test_iou_closed, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.optimizer_params)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            **self.scheduler_params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val_miou",
                "frequency": 1,
            },
        }
