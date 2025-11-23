from torchvision.transforms import v2
import torch


def get_transforms(cfg: dict):
    image_size = (cfg["img_height"], cfg["img_width"])
    mean = cfg["img_mean"]
    std = cfg["img_std"]
    use_augmentation = cfg.get("use_augmentation", True)

    base_transforms = [
        v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    train_transforms = base_transforms.copy()

    if use_augmentation:
        train_transforms.insert(1, v2.RandomHorizontalFlip(p=0.5))

    val_transforms = v2.Compose(base_transforms)
    train_transforms = v2.Compose(train_transforms)

    return train_transforms, val_transforms
