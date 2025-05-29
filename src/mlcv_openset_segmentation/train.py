import logging
from pathlib import Path

import torch
import wandb
import yaml

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from .datamodule import StreetHazardsDataModule
from .model_uncertainty import UncertaintyModel
from .transforms import get_transforms

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run = wandb.init(project="mlcv-assignment")
    run_id = run.id
    logger = WandbLogger(experiment=run)

    L.seed_everything(cfg["seed"], workers=True)

    train_transform, eval_transform = get_transforms(cfg["data"])

    data_module = StreetHazardsDataModule(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    model = UncertaintyModel(
        num_classes=cfg["data"]["num_classes"],
        model_name=cfg["model"]["model_name"],
        use_aux_loss=cfg["model"]["use_aux_loss"],
        uncertainty_type=cfg["model"]["uncertainty_type"],
        optimizer_kwargs=cfg["optimizer"],
        scheduler_kwargs=cfg["scheduler"],
    )

    save_dir = Path("checkpoints/")
    save_dir /= run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = L.Trainer(
        logger=logger,
        callbacks=[model_checkpoint, early_stopping],
        **cfg["trainer"],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()

if __name__ == "__main__":
    main()
