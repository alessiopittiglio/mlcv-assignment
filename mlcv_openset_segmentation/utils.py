from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mlcv_openset_segmentation.visualize import COLORS


def unnormalize_image(image, mean, std):
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    image = image * std_tensor + mean_tensor
    return image.clamp(0, 1)


def colorize_mask(mask, colors):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        colored_mask[mask == (class_id + 1)] = color

    return colored_mask


def remap_mask(mask, mapping):
    remapped = torch.full_like(mask, fill_value=255)

    for src, dst in mapping.items():
        remapped[mask == src] = dst

    return remapped


def visualize_mask(prediction, label_mapping):
    mapped_mask = remap_mask(prediction, label_mapping)
    mapped_mask_np = mapped_mask.cpu().numpy()

    return colorize_mask(mapped_mask_np, COLORS)


def collect_one_sample_per_class(dataloader, num_classes):
    samples = {}

    for images, masks in dataloader:
        for image, mask in zip(images, masks):
            for class_id in range(num_classes):
                if class_id not in samples and (mask == class_id).any():
                    samples[class_id] = (image.clone(), mask.clone())

            if len(samples) == num_classes:
                return samples

    return samples


def visualize_samples_per_class(
    samples,
    class_names,
    image_mean,
    image_std,
    alpha=0.6,
    cols=4,
):
    num_classes = len(class_names)
    rows = ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = np.array(axes).flatten()

    for class_id, ax in enumerate(axes[:num_classes]):
        image, mask = samples[class_id]

        image = unnormalize_image(image, image_mean, image_std)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()

        class_mask = mask_np == class_id

        overlay = np.zeros((*class_mask.shape, 4))
        overlay[..., 0] = 1.0
        overlay[..., 3] = class_mask * alpha

        ax.imshow(image_np)
        ax.imshow(overlay)
        ax.set_title(class_names[class_id])
        ax.axis("off")

    for ax in axes[num_classes:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compute_pixel_frequencies(dataloader, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for _, masks in tqdm(dataloader, desc="Counting pixels per class"):
        masks_np = masks.cpu().numpy()

        for class_id in range(num_classes):
            class_counts[class_id] += np.count_nonzero(masks_np == class_id)

        total_pixels += masks_np.size

    frequencies = class_counts / total_pixels
    return class_counts, frequencies


def get_or_compute_frequencies(dataloader, num_classes, output_path):
    output_path = Path(output_path)

    if output_path.exists():
        data = np.loadtxt(output_path, delimiter=",", skiprows=1)
        class_counts = data[:, 1].astype(np.int64)
        class_frequencies = data[:, 2].astype(np.float64)
        return class_counts, class_frequencies

    class_counts, class_frequencies = compute_pixel_frequencies(
        dataloader,
        num_classes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "class_id,class_count,class_frequency"
    data = np.column_stack((np.arange(num_classes), class_counts, class_frequencies))

    np.savetxt(
        output_path,
        data,
        fmt=["%d", "%d", "%.8f"],
        delimiter=",",
        header=header,
        comments="",
    )

    return class_counts, class_frequencies


def plot_class_distribution(
    class_names,
    train_frequencies,
    val_frequencies,
):
    class_indices = np.arange(len(class_names))
    train_percent = train_frequencies * 100
    val_percent = val_frequencies * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    def plot_side(ax, values, title, color=None):
        bars = ax.bar(class_indices, values, color=color)
        ax.set_title(title)
        ax.set_ylabel("Percentage of Pixels (%)")
        ax.set_xticks(class_indices)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.bar_label(bars, fmt="%.2f%%")

    plot_side(
        axes[0], train_percent, "Class distribution on StreetHazards training set"
    )
    plot_side(
        axes[1],
        val_percent,
        "Class distribution on StreetHazards validation set",
        color="darkorange",
    )

    fig.suptitle("Class distribution")
    fig.tight_layout()
    plt.show()


def compute_global_overlap(stats):
    global_fpr = 0

    overlaps_low = []
    overlaps_high = []

    for class_id, class_stats in stats.items():
        if not isinstance(class_id, int):
            continue

        id_q1 = class_stats.get("ID_Q1")
        id_q3 = class_stats.get("ID_Q3")
        ood_q1 = class_stats.get("OOD_Q1")
        ood_q3 = class_stats.get("OOD_Q3")

        if None in (id_q1, id_q3, ood_q1, ood_q3):
            continue

        overlap_low = max(id_q1, ood_q1)
        overlap_high = min(id_q3, ood_q3)

        if overlap_high > overlap_low:
            overlaps_low.append(overlap_low)
            overlaps_high.append(overlap_high)

    if not overlaps_low:
        return None, None, global_fpr

    return min(overlaps_low), max(overlaps_high), global_fpr
