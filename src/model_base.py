import torch
import torch.nn as nn
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
            num_classes_known=12,
            learning_rate=1e-4,
            use_aux_loss=False,
            model_name='deeplabv3_resnet50',
            pretrained_weights='DEFAULT',
        ):
        super().__init__()
        self.save_hyperparameters()

        model_fn = getattr(segmentation, model_name, None)
        self.model = model_fn(weights=pretrained_weights)

        def replace_classifier(module, classes):
            for i in range(len(module) - 1, -1, -1):
                if isinstance(module[i], nn.Conv2d):
                    conv = module[i]
                    module[i] = nn.Conv2d(
                        conv.in_channels, 
                        classes, 
                        kernel_size=conv.kernel_size, 
                        stride=conv.stride
                    )
                    return module
            raise TypeError(f"No Conv2d layer found in classifier")
        
        if hasattr(self.model, 'classifier'):
            self.model.classifier = replace_classifier(
                self.model.classifier, num_classes_known
            )

        if use_aux_loss and hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier = replace_classifier(
                self.model.aux_classifier, num_classes_known
            )
        elif not hasattr(self.model, 'aux_classifier'):
            self.hparams.use_aux_loss = False

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        
        miou_args = dict(
            task="multiclass",
            num_classes=num_classes_known,
            average='macro',
            ignore_index=IGNORE_INDEX,
        )
        self.train_miou = torchmetrics.JaccardIndex(**miou_args)
        self.val_miou = torchmetrics.JaccardIndex(**miou_args) 
        self.test_miou_closed = torchmetrics.JaccardIndex(
            task="multiclass",
            average='macro',
            num_classes=num_classes_known+1, 
            ignore_index=ANOMALY_ID,
        )

    def _calculate_segmentation_loss(self, outputs, masks):
        loss = self.criterion(outputs['out'], masks)

        if self.hparams.use_aux_loss and 'aux' in outputs:
            aux_preds = outputs['aux']
            if aux_preds.shape[-2:] != masks.shape[-2:]:
                aux_preds = nn.functional.interpolate(
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
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

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
