import logging
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics import JaccardIndex
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from mlcv_openset_segmentation.visualize import COLORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_COLOR = "tab:orange"
OOD_COLOR = "tab:blue"
OVERLAP_COLOR = "gray"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def unnormalize_image(image, mean, std):
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    image = image * std_tensor + mean_tensor
    return image.clamp(0, 1)


def visualize_mask(prediction):
    mask = prediction.cpu().numpy().astype(int)
    return COLORS[mask]


# ---------------------------------------------------------------------------
# Dataset Scanning
# ---------------------------------------------------------------------------


def collect_representative_samples(dataloader, num_classes, max_batches=5):
    samples = {}
    max_pixels = {cls: 0 for cls in range(num_classes)}

    for batch_idx, (images, masks) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        for img, mask in zip(images, masks):
            for cls_id in torch.unique(mask).tolist():
                if not 0 <= cls_id < num_classes:
                    continue

                count = int((mask == cls_id).sum())

                if count > max_pixels[cls_id]:
                    max_pixels[cls_id] = count
                    samples[cls_id] = (img.clone(), mask.clone())

    return dict(sorted(samples.items()))


def compute_pixel_frequencies(dataloader, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for _, masks in tqdm(dataloader, desc="Computing class frequencies"):
        masks_np = masks.cpu().numpy()
        total_pixels += masks_np.size

        for cls_id in range(num_classes):
            class_counts[cls_id] += np.count_nonzero(masks_np == cls_id)

    frequencies = class_counts / total_pixels
    return class_counts, frequencies


def load_or_compute_frequencies(dataloader, num_classes, output_path):
    output_path = Path(output_path)

    if output_path.exists():
        data = np.loadtxt(output_path, delimiter=",", skiprows=1)
        counts = data[:, 1].astype(np.int64)
        freqs = data[:, 2].astype(np.float64)
        return counts, freqs

    counts, freqs = compute_pixel_frequencies(dataloader, num_classes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((np.arange(num_classes), counts, freqs))

    np.savetxt(
        output_path,
        data,
        fmt=["%d", "%d", "%.8f"],
        delimiter=",",
        header="class_id,class_count,class_frequency",
        comments="",
    )

    return counts, freqs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


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
    axes = axes.flatten()

    for cls_id, ax in enumerate(axes[:num_classes]):
        image, mask = samples[cls_id]

        image = unnormalize_image(image, image_mean, image_std)
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        class_mask = mask_np == cls_id
        overlay = np.zeros((*class_mask.shape, 4))
        overlay[..., 0] = 1.0  # red overlay
        overlay[..., 3] = class_mask * alpha

        ax.imshow(image_np)
        ax.imshow(overlay)
        ax.set_title(class_names[cls_id])
        ax.axis("off")

    for ax in axes[num_classes:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    class_names,
    train_freq,
    val_freq,
):
    indices = np.arange(len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, data, title, color in zip(
        axes,
        [train_freq, val_freq],
        ["Training Set", "Validation Set"],
        ["tab:blue", "tab:orange"],
    ):
        bars = ax.bar(indices, data * 100, color=color)
        ax.set_title(title)
        ax.set_xticks(indices)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylabel("Percentage of Pixels (%)")
        ax.bar_label(bars, fmt="%.2f%%")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Metrics and Logging
# ---------------------------------------------------------------------------


def smooth_curve(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def load_run_metrics(run_name, logs_dir="logs", version=None):
    run_path = Path(logs_dir) / run_name
    if not run_path.exists():
        logger.warning("Run not found: %s", run_path)
        return None

    versions = sorted(run_path.glob("version_*"))

    target = versions[-1] if version is None else run_path / f"version_{version}"
    csv_path = target / "metrics.csv"

    if not csv_path.exists():
        logger.warning("Metrics file missing", csv_path)
        return None

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        logger.warning("'epoch' column missing in %s", csv_path)
        return None

    return df.groupby("epoch", as_index=False).mean()


def get_best_metric(run_name, metric, mode="max"):
    df = load_run_metrics(run_name)
    if df is None or metric not in df:
        return None

    if mode == "max":
        return df[metric].max()
    elif mode == "min":
        return df[metric].min()
    else:
        raise ValueError("mode must be 'max' or 'min'")


# ---------------------------------------------------------------------------
# Statistics Plotting
# ---------------------------------------------------------------------------


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


def plot_class_statistics(ax, stats, class_names, title):
    num_classes = len(class_names)
    x_positions = np.arange(num_classes)
    x_min, x_max = -0.5, num_classes - 0.5

    global_low, global_high, global_fpr = compute_global_overlap(stats)

    for class_id, class_name in enumerate(class_names):
        class_stats = stats[class_id]

        id_q1, id_q3 = class_stats["ID_Q1"], class_stats["ID_Q3"]
        ood_q1, ood_q3 = class_stats["OOD_Q1"], class_stats["OOD_Q3"]
        id_mean, ood_mean = class_stats["ID_mean"], class_stats["OOD_mean"]

        ax.vlines(class_id, id_q1, id_q3, colors=ID_COLOR, lw=4, zorder=2)
        ax.scatter(class_id, id_mean, color=ID_COLOR, s=40, zorder=3)

        if ood_q1 is not None and ood_q3 is not None:
            ax.vlines(class_id, ood_q1, ood_q3, colors=OOD_COLOR, lw=4, zorder=2)
        if ood_mean is not None:
            ax.scatter(class_id, ood_mean, color=OOD_COLOR, s=40, zorder=3)

    if global_low is not None and global_high > global_low:
        ax.fill_between(
            [x_min, x_max],
            global_low,
            global_high,
            color=OVERLAP_COLOR,
            alpha=min(global_fpr, 1.0) * 0.3,
            zorder=1,
        )

    ax.set_xlabel(title, labelpad=15, fontsize=12, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")


def create_statistics_plots(stats_path, class_names):
    stats = torch.load(stats_path, weights_only=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    plots = [
        ("quantiles_msp", "Max Softmax Probability"),
        ("quantiles_logit", "Max Logit"),
        ("quantiles_sml", "Standardized Max Logit"),
    ]

    for ax, (key, title) in zip(axes, plots):
        plot_class_statistics(ax, stats[key], class_names, title)

    legend_elements = [
        Line2D([0], [0], color=ID_COLOR, lw=3, label="In-distribution"),
        Line2D([0], [0], color=OOD_COLOR, lw=3, label="Out-of-distribution"),
        Line2D([0], [0], color=OVERLAP_COLOR, lw=3, label="Overlap", alpha=0.5),
    ]

    axes[0].legend(handles=legend_elements, loc="lower left", frameon=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training Curve Plotting
# ---------------------------------------------------------------------------


def plot_metrics_dual_axis(
    runs,
    metrics_left=("train_loss", "val_loss"),
    metrics_right=("train_miou", "val_miou"),
    labels=None,
    smooth=False,
    smooth_window=3,
    title_left=None,
    title_right=None,
):
    if labels is None:
        labels = {}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for run in runs:
        df = load_run_metrics(run)
        if df is None:
            continue

        for metric in metrics_left:
            if metric in df:
                if smooth:
                    y = smooth_curve(df[metric], smooth_window)

                label_name = labels.get(metric, metric)
                ax_left.plot(df["epoch"], y, label=label_name)

        for metric in metrics_right:
            if metric in df:
                if smooth:
                    y = smooth_curve(df[metric], smooth_window)

                label_name = labels.get(metric, metric)
                ax_right.plot(df["epoch"], y, label=label_name)

    ax_left.set_title(title_left)
    ax_right.set_title(title_right)

    for ax in (ax_left, ax_right):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_run(
    runs,
    labels=None,
    metrics=None,
    logs_dir="logs",
    smooth=False,
    smooth_window=3,
    figsize=(12, 6),
    title=None,
):
    if labels is None:
        labels = runs

    if len(runs) != len(labels):
        raise ValueError("runs and labels must have the same length")

    plt.figure(figsize=figsize)

    for run, label in zip(runs, labels):
        df = load_run_metrics(run, logs_dir)
        if df is None:
            continue

        for key, name in metrics:
            if key not in df:
                continue

            if smooth:
                y = smooth_curve(df[key], smooth_window)

            plt.plot(df["epoch"], y, label=f"{label}")

    if title:
        plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runs_dual_axis(
    runs,
    metrics_left=[("train_loss", "Train Loss"), ("val_loss", "Val Loss")],
    metrics_right=[("train_miou", "Train mIoU"), ("val_miou", "Val mIoU")],
    labels=None,
    smooth=False,
    smooth_window=3,
    title_left=None,
    title_right=None,
    logs_dir="logs",
):
    if labels is None:
        labels = runs

    if len(runs) != len(labels):
        raise ValueError("runs and labels must have the same length")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for run, label in zip(runs, labels):
        df = load_run_metrics(run, logs_dir=logs_dir)
        if df is None or df.empty:
            continue

        for metric_key, metric_name in metrics_left:
            if metric_key not in df:
                continue

            y = df[metric_key]
            if smooth:
                y = smooth_curve(y, window=smooth_window)

            ax_left.plot(df["epoch"], y, label=f"{label}")

        for metric_key, metric_name in metrics_right:
            if metric_key not in df:
                continue

            y = df[metric_key]
            if smooth:
                y = smooth_curve(y, window=smooth_window)

            ax_right.plot(df["epoch"], y, label=f"{label}")

    if title_left is not None:
        ax_left.set_title(title_left)

    if title_right is not None:
        ax_right.set_title(title_right)

    for ax in (ax_left, ax_right):
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="-", alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    ax_left.set_ylabel("Loss")
    ax_right.set_ylabel("Metric Value")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# IoU Computation
# ---------------------------------------------------------------------------


def compute_per_class_iou(model, dataloader, num_classes, ignore_index, device):
    metric = JaccardIndex(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=None,
    ).to(device)

    model.eval()
    for images, masks in tqdm(dataloader, desc="Computing per-class IoU"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        metric.update(preds, masks)

    return metric.compute().cpu().numpy()


def load_or_compute_per_class_iou(
    model, dataloader, class_names, ignore_index, device, csv_path
):
    csv_path = Path(csv_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        iou_values = (df["IoU (%)"].values / 100.0).astype(float)
        return iou_values, df

    iou_values = compute_per_class_iou(
        model,
        dataloader,
        num_classes=len(class_names),
        ignore_index=ignore_index,
        device=device,
    )

    df = pd.DataFrame(
        {
            "Class": class_names[: len(iou_values)],
            "IoU (%)": np.round(iou_values * 100, 2),
        }
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    return iou_values, df
