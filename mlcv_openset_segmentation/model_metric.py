import numpy as np
import torch
import lightning as L
from torchmetrics import JaccardIndex
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from mlcv_openset_segmentation.dmlnet import ModelBuilder

ANOMALY_ID = 13
IGNORE_INDEX = 255


def compute_ood_metrics(out_scores, in_scores):
    labels = np.concatenate([np.ones_like(out_scores), np.zeros_like(in_scores)])
    scores = np.concatenate([out_scores, in_scores])
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr_at_95_tpr = fpr[np.abs(tpr - 0.95).argmin()]
    return auroc, aupr, fpr_at_95_tpr


def evaluate_ood(confidence, segmentation):
    ood_mask = segmentation == ANOMALY_ID

    in_scores = -confidence[~ood_mask]
    out_scores = -confidence[ood_mask]

    if out_scores.size == 0 or in_scores.size == 0:
        return None

    return compute_ood_metrics(out_scores, in_scores)


def normalize_array(x):
    min_val = x.min()
    max_val = x.max()

    if max_val == min_val:
        return x - min_val

    return (x - min_val) / (max_val - min_val)


class MetricLearningModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 13,
        encoder_path: str = None,
        decoder_path: str = None,
        in_channels: int = 2048,
        max_confidence_clip: float = 400.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ModelBuilder.build_encoder(weights_path=encoder_path)
        self.decoder = ModelBuilder.build_decoder(
            in_channels=in_channels, num_classes=num_classes, weights_path=decoder_path
        )

        self.test_miou = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=IGNORE_INDEX,
        )

        self.ood_results = []

    def forward(self, images, seg_size=None):
        features = self.encoder(images, return_feature_maps=True)
        return self.decoder(features, seg_size=seg_size)

    def test_step(self, batch, batch_idx):
        images, seg_labels = batch
        seg_np = seg_labels.squeeze().cpu().numpy()

        self.decoder.use_softmax = True
        scores, _ = self(images, seg_size=seg_labels.shape[-2:])
        self.decoder.use_softmax = False

        scores = scores.squeeze(0)

        # Dissum OOD method
        dissimilarity = torch.sum(scores, dim=0)
        dissimilarity = (-dissimilarity).cpu().numpy()
        dissimilarity = np.clip(dissimilarity, None, self.hparams.max_confidence_clip)

        confidence = normalize_array(dissimilarity)

        ood_metrics = evaluate_ood(confidence, seg_np)
        if ood_metrics is not None:
            auroc, aupr, fpr = ood_metrics
            self.ood_results.append({"auroc": auroc, "aupr": aupr, "fpr": fpr})

        predictions = torch.argmax(scores, dim=0, keepdim=True)

        seg_for_iou = seg_labels.clone()
        seg_for_iou[seg_for_iou == ANOMALY_ID] = IGNORE_INDEX

        self.test_miou.update(predictions, seg_for_iou)
        self.log("test_miou", self.test_miou, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        miou = self.test_miou.compute().item()

        print("\n" + "=" * 50)
        print("[Evaluation Summary]:")
        print(f"Mean IoU: {miou:.4f}")

        if self.ood_results:
            aurocs = [r["auroc"] for r in self.ood_results]
            auprs = [r["aupr"] for r in self.ood_results]
            fprs = [r["fpr"] for r in self.ood_results]

            print(f"Mean AUROC: {np.mean(aurocs):.4f}")
            print(f"Mean AUPR: {np.mean(auprs):.4f}")
            print(f"Mean FPR@95TPR: {np.mean(fprs):.4f}")

        print("=" * 50 + "\n")

        self.ood_results.clear()
        self.test_miou.reset()
