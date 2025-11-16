import argparse
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from mlcv_openset_segmentation.datamodule import StreetHazardsDataModule
from mlcv_openset_segmentation.model_metric import (
    MetricLearningModel,
    IGNORE_INDEX,
    ANOMALY_ID,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings using t-SNE.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="tsne_plot.png",
        help="Path to save the t-SNE plot",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(
    model,
    dataloader,
    device,
    num_samples_per_class=100,
    max_anomaly_samples=200,
    per_batch_samples=50,
):
    """
    Extract prototypes and per-pixel embeddings from a trained model.
    """
    model.to(device)
    model.eval()

    prototypes = model.prototypes.detach().cpu().numpy()

    all_known_embeddings, all_known_labels, all_anomaly_embeddings = [], [], []

    print("Extracting embeddings from dataloader...")
    for images, gt_masks in tqdm(dataloader, desc="Processing batches"):
        images, gt_masks = images.to(device), gt_masks.to(device)
        embeddings = model(images)  # [N, D, H, W]

        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(
            -1, model.hparams.embedding_dim
        )
        masks_flat = gt_masks.reshape(-1)

        valid_mask = masks_flat != IGNORE_INDEX
        embeddings_valid = embeddings_flat[valid_mask]
        masks_valid = masks_flat[valid_mask]

        if len(embeddings_valid) > per_batch_samples:
            idx = torch.randperm(len(embeddings_valid))[:per_batch_samples]
            embeddings_valid = embeddings_valid[idx]
            masks_valid = masks_valid[idx]

        known_mask = masks_valid != ANOMALY_ID
        anomaly_mask = masks_valid == ANOMALY_ID

        all_known_embeddings.append(embeddings_valid[known_mask].cpu())
        all_known_labels.append(masks_valid[known_mask].cpu())
        all_anomaly_embeddings.append(embeddings_valid[anomaly_mask].cpu())

        del embeddings, outputs, images, gt_masks
        torch.cuda.empty_cache()

    # Concatenate results
    known_embeddings = torch.cat(all_known_embeddings, dim=0)
    known_labels = torch.cat(all_known_labels, dim=0)
    anomaly_embeddings = torch.cat(all_anomaly_embeddings, dim=0)

    print(
        f"Found {len(known_embeddings)} known embeddings and {len(anomaly_embeddings)} anomalies."
    )
    print(f"Sampling up to {num_samples_per_class} embeddings per class...")

    # Sample known embeddings uniformly per class
    sampled_embeddings, sampled_labels = [], []
    for class_id in range(model.hparams.num_classes):
        class_mask = known_labels == class_id
        class_embeddings = known_embeddings[class_mask]
        if len(class_embeddings) == 0:
            continue
        if len(class_embeddings) > num_samples_per_class:
            indices = torch.randperm(len(class_embeddings))[:num_samples_per_class]
            class_embeddings = class_embeddings[indices]
        sampled_embeddings.append(class_embeddings)
        sampled_labels.extend([class_id] * len(class_embeddings))

    final_known_embeddings = torch.cat(sampled_embeddings, dim=0).numpy()
    final_known_labels = np.array(sampled_labels)

    if len(anomaly_embeddings) > max_anomaly_samples:
        idx = torch.randperm(len(anomaly_embeddings))[:max_anomaly_samples]
        anomaly_embeddings = anomaly_embeddings[idx]

    final_anomaly_embeddings = anomaly_embeddings.numpy()

    return (
        prototypes,
        final_known_embeddings,
        final_known_labels,
        final_anomaly_embeddings,
    )


def visualize_with_tsne(
    prototypes,
    known_embeddings,
    known_labels,
    anomaly_embeddings,
    num_classes,
    output_path="tsne_plot.png",
):
    """Visualize embeddings and anomalies using t-SNE."""
    print("Combining all points for t-SNE projection...")
    all_points = np.vstack([prototypes, known_embeddings, anomaly_embeddings])

    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    all_points_2d = tsne.fit_transform(all_points)

    num_known = len(known_embeddings)
    known_2d = all_points_2d[:num_known]
    anomaly_2d = all_points_2d[num_known:]

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    palette = sns.color_palette("bright", num_classes)

    x_min, x_max = all_points_2d[:, 0].min(), all_points_2d[:, 0].max()
    y_min, y_max = all_points_2d[:, 1].min(), all_points_2d[:, 1].max()

    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Known embeddings
    for i in range(num_classes):
        mask = known_labels == i
        for x, y in known_2d[mask]:
            plt.text(
                x,
                y,
                str(i),
                color=palette[i],
                fontweight="bold",
                fontsize=9,
                ha="center",
                va="center",
                alpha=0.8,
            )

    # Anomalies
    if len(anomaly_2d) > 0:
        plt.scatter(
            anomaly_2d[:, 0],
            anomaly_2d[:, 1],
            color="black",
            marker="x",
            label="Anomaly",
            alpha=0.8,
            s=30,
        )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading model from: {args.checkpoint_path}")
    model = MetricLearningModel.load_from_checkpoint(args.checkpoint_path)

    test_data_module = StreetHazardsDataModule(
        root_dir="data/",
        batch_size=4,
        num_workers=4,
    )
    test_data_module.setup(stage="test")
    test_loader = test_data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prototypes, known_emb, known_lbl, anomaly_emb = extract_embeddings(
        model, test_loader, device, num_samples_per_class=10
    )

    visualize_with_tsne(
        prototypes,
        known_emb,
        known_lbl,
        anomaly_emb,
        model.hparams.num_classes,
        output_path=args.output_path,
    )
