import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.segmentation as segmentation
import lightning as L
import torchmetrics

IGNORE_INDEX = 255
ANOMALY_ID = 12

class BaseSemanticSegmentationModel(L.LightningModule):
    """
    Base class for semantic segmentation models using PyTorch Lightning.
    """
    def __init__(
            self,
            num_classes: int = 12,
            model_name: str = 'deeplabv3_resnet50',
            use_aux_loss: bool = True,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._init_model(
            self.hparams.model_name,
            self.hparams.num_classes,
            self.hparams.use_aux_loss,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        
        miou_args = dict(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average='macro',
            ignore_index=IGNORE_INDEX,
        )
        self.train_miou = torchmetrics.JaccardIndex(**miou_args)
        self.val_miou = torchmetrics.JaccardIndex(**miou_args)
        
        self.test_miou_closed = torchmetrics.JaccardIndex(
            task="multiclass",
            average='macro',
            num_classes=self.hparams.num_classes+1, 
            ignore_index=ANOMALY_ID,
        )
    
    def _init_model(
            self,
            model_name: str,
            num_classes: int,
            use_aux_loss: bool
        ) -> nn.Module:
        if model_name == 'deeplabv3_resnet50':
            weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            model = segmentation.deeplabv3_resnet50(weights=weights)
        elif model_name == 'deeplabv3_resnet101':
            weights = segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
            model = segmentation.deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model.classifier[-1] = self._replace_classifier_head(
            model.classifier[-1], num_classes
        )

        if use_aux_loss:
            model.aux_classifier[-1] = self._replace_classifier_head(
                model.aux_classifier[-1],
                num_classes
            )
        
        return model
    
    @staticmethod
    def _replace_classifier_head(conv_layer: nn.Conv2d, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
        )

    def _calculate_segmentation_loss(self, outputs, masks):
        loss = self.criterion(outputs['out'], masks)

        if self.hparams.use_aux_loss and 'aux' in outputs:
            aux_preds = outputs['aux']
            if aux_preds.shape[-2:] != masks.shape[-2:]:
                aux_preds = F.interpolate(
                    aux_preds, 
                    size=masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            aux_loss = self.criterion(aux_preds, masks)
            loss += 0.4 * aux_loss
        return loss

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, torch.Tensor):
            outputs = {'out': outputs}
        return outputs

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        logits = outputs['out']

        loss = self._calculate_segmentation_loss(outputs, masks)
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        preds = torch.argmax(logits, dim=1)
        self.train_miou.update(preds, masks)
        self.log(
            'train_miou',
            self.train_miou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        logits = outputs['out']

        loss = self._calculate_segmentation_loss(outputs, masks)
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        preds = torch.argmax(logits, dim=1)
        self.val_miou.update(preds, masks)
        self.log(
            'val_miou',
            self.val_miou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def test_step(self, batch, batch_idx):
        images, masks_gt = batch
        outputs = self(images)
        logits_known = outputs['out']

        preds_closed = torch.argmax(logits_known, dim=1)
        self.test_miou_closed.update(preds_closed, masks_gt)
        self.log(
            'test_miou_closed',
            self.test_miou_closed,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_kwargs)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self.hparams.scheduler_kwargs,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }
    
if __name__ == "__main__":
    model = BaseSemanticSegmentationModel(
        model_name='deeplabv3_resnet50',
        use_aux_loss=True,
    )
    print(model)
