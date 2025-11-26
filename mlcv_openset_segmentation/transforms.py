from torchvision.transforms import v2
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STREET_HAZARD_MEAN = [0.3301, 0.3458, 0.3729]
STREET_HAZARD_STD = [0.1616, 0.1600, 0.1741]


def get_transforms(cfg: dict):
    image_size = (cfg["img_height"], cfg["img_width"])

    data_cfg = cfg.get("data", {})
    aug_cfg = data_cfg.get("augmentation", {})

    normalization_type = data_cfg.get("normalization", "image_net")

    if normalization_type == "image_net":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization_type == "street_hazard":
        mean, std = STREET_HAZARD_MEAN, STREET_HAZARD_STD
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")

    use_augmentation = data_cfg.get("use_augmentation", True)
    use_color_jitter = aug_cfg.get("color_jitter", False)
    use_random_crop = aug_cfg.get("random_resized_crop", False)
    use_horizontal_flip = aug_cfg.get("horizontal_flip", True)

    scale_min = aug_cfg.get("scale_min", 0.8)
    scale_max = aug_cfg.get("scale_max", 1.0)

    resize_transform = (
        v2.RandomResizedCrop(
            size=image_size,
            scale=(scale_min, scale_max),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        if use_random_crop
        else v2.Resize(
            size=image_size,
            interpolation=v2.InterpolationMode.BILINEAR,
        )
    )

    base_transforms = [
        resize_transform,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    train_transforms = list(base_transforms)

    if use_augmentation:
        augmentation_transforms = []

        if use_horizontal_flip:
            augmentation_transforms.append(v2.RandomHorizontalFlip(p=0.5))

        if use_color_jitter:
            augmentation_transforms.append(
                v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                )
            )

        train_transforms = (
            train_transforms[:1] + augmentation_transforms + train_transforms[1:]
        )

    train_transforms = v2.Compose(train_transforms)
    val_transforms = v2.Compose(base_transforms)
    normalize_only = v2.Normalize(mean=mean, std=std)

    return train_transforms, val_transforms, normalize_only
