import lightning as L
from torch.utils.data import DataLoader

from mlcv_openset_segmentation.datasets.dataset import StreetHazardsDataset
from mlcv_openset_segmentation.datasets.dataset_oe import StreetHazardsOEDataset


class StreetHazardsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        eval_transform=None,
        anomaly_normalize=None,
        outlier_val_dataset=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.anomaly_normalize = anomaly_normalize
        self.outlier_dataset_val = outlier_val_dataset

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = StreetHazardsDataset(
                root_dir=self.root_dir, split="train", transform=self.train_transform
            )
            self.val_dataset = StreetHazardsDataset(
                root_dir=self.root_dir, split="val", transform=self.eval_transform
            )
        if stage == "test" or stage is None:
            base_val = StreetHazardsDataset(
                root_dir=self.root_dir, split="val", transform=self.eval_transform
            )

            self.test_dataset = StreetHazardsOEDataset(
                base_dataset=base_val,
                outlier_dataset=self.outlier_dataset_val,
                inject_probability=1.0,
                max_anomalies_per_image=3,
                transform=self.anomaly_normalize,
                deterministic=True,
                seed=42,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class StreetHazardsOEDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir,
        outlier_train_dataset,
        outlier_val_dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform=None,
        eval_transform=None,
        anomaly_normalize=None,
        inject_probability: float = 0.5,
        max_anomalies_per_image: int = 3,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.outlier_train_dataset = outlier_train_dataset
        self.outlier_val_dataset = outlier_val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.anomaly_normalize = anomaly_normalize
        self.inject_probability = inject_probability
        self.max_anomalies_per_image = max_anomalies_per_image

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            base_train = StreetHazardsDataset(
                root_dir=self.root_dir, split="train", transform=self.train_transform
            )
            self.train_dataset = StreetHazardsOEDataset(
                base_train,
                self.outlier_train_dataset,
                inject_probability=self.inject_probability,
                max_anomalies_per_image=self.max_anomalies_per_image,
                transform=self.anomaly_normalize
            )

            base_val = StreetHazardsDataset(
                root_dir=self.root_dir, split="val", transform=self.eval_transform
            )

            self.val_dataset = StreetHazardsOEDataset(
                base_dataset=base_val,
                outlier_dataset=self.outlier_val_dataset,
                inject_probability=1.0,
                max_anomalies_per_image=3,
                transform=self.anomaly_normalize,
                deterministic=True,
                seed=42,
            )

        if stage == "test" or stage is None:
            self.test_dataset = StreetHazardsDataset(
                root_dir=self.root_dir, split="test", transform=self.eval_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


if __name__ == "__main__":
    from pathlib import Path

    data_module = StreetHazardsDataModule(
        root_dir=Path("data/"), batch_size=32, num_workers=0
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for batch in train_loader:
        images, masks = batch
        print(images.shape, masks.shape)
        break
