import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from mlcv_openset_segmentation.dataset import StreetHazardsDataset


class StreetHazardsOEDataset(torch.utils.data.Dataset):
    """
    Outlier-Exposure wrapper for StreetHazards.
    """

    def __init__(
        self,
        base_dataset: StreetHazardsDataset,
        outlier_dataset,
        inject_probability: float = 0.3,
        max_anomalies_per_image: int = 2,
        transform=None,
        deterministic: bool = False,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.outlier_dataset = outlier_dataset
        self.inject_probability = inject_probability
        self.max_anomalies_per_image = max_anomalies_per_image
        self.anomaly_id = StreetHazardsDataset.ANOMALY_ID
        self.transform = transform
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.deterministic:
            self._set_seed(idx)

        image, mask = self.base_dataset[idx]

        if random.random() > self.inject_probability:
            return image, mask

        num_anomalies = random.randint(1, self.max_anomalies_per_image)
        for _ in range(num_anomalies):
            image, mask = self._inject_anomaly(image, mask)

        return image, mask

    def _set_seed(self, idx):
        seed = self.seed + idx
        random.seed(seed)
        np.random.seed(seed)

    def _inject_anomaly(self, image, mask):
        _, height, width = image.shape

        crop_h = np.random.randint(int(0.1 * height), int(0.3 * height) + 1)
        crop_w = np.random.randint(int(0.1 * width), int(0.3 * width) + 1)

        top, left, crop_h, crop_w = transforms.RandomCrop.get_params(
            image, output_size=(crop_h, crop_w)
        )

        anomaly_img, anomaly_mask, anomaly_class = self._sample_valid_anomaly()

        anomaly_img = F.interpolate(
            anomaly_img.unsqueeze(0),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        anomaly_mask = (
            F.interpolate(
                anomaly_mask.unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode="nearest",
            )
            .squeeze(0)
            .long()
        )

        region_mask = anomaly_mask.squeeze(0) == anomaly_class

        patch_img = image[:, top : top + crop_h, left : left + crop_w]
        patch_img[:, region_mask] = anomaly_img[:, region_mask]

        patch_mask = mask[top : top + crop_h, left : left + crop_w]
        patch_mask[region_mask] = self.anomaly_id

        return image, mask

    def _sample_valid_anomaly(self, max_attempts: int = 50):
        for _ in range(max_attempts):
            idx = np.random.randint(0, len(self.outlier_dataset))
            anomaly_image, anomaly_mask = self.outlier_dataset[idx]

            anomaly_image = transforms.ToTensor()(anomaly_image)
            if self.transform is not None:
                anomaly_image = self.transform(anomaly_image)

            anomaly_mask = torch.from_numpy(
                np.array(anomaly_mask, dtype=np.int64)
            ).unsqueeze(0)

            unique_classes = np.unique(anomaly_mask.cpu().numpy())
            valid_classes = unique_classes[
                (unique_classes != 0) & (unique_classes != 255)
            ]

            if len(valid_classes) > 0:
                anomaly_class = int(np.random.choice(valid_classes))
                return anomaly_image, anomaly_mask, anomaly_class

        raise RuntimeError(
            "No valid anomaly class found in outlier dataset after "
            f"{max_attempts} attempts."
        )


if __name__ == "__main__":
    from pathlib import Path
    from torchvision.datasets import VOCSegmentation

    voc_dataset = VOCSegmentation(
        root="data/VOC",
        year="2012",
        image_set="train",
    )
    print(f"VOC train dataset size: {len(voc_dataset)}")

    base_dataset = StreetHazardsDataset(
        root_dir=Path("data/"),
        split="train",
    )

    oe_dataset = StreetHazardsOEDataset(
        base_dataset=base_dataset,
        outlier_dataset=voc_dataset,
        inject_probability=0.5,
        max_anomalies_per_image=3,
    )
