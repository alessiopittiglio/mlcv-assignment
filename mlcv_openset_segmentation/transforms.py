from torchvision.transforms import v2
import torch


def get_transforms(cfg: dict):
    transform = v2.Compose(
        [
            v2.Resize(
                (cfg["img_height"], cfg["img_width"]),
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg["img_mean"], std=cfg["img_std"]),
        ]
    )
    return transform, transform
