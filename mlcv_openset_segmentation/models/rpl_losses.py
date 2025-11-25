import torch
import torch.nn.functional as F


def disimilarity_entropy(logits, vanilla_logits, t=1.0):
    """
    Dissimilarity entropy regularization.

    Adapted from:
    Liu et al., "Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection
    in Semantic Segmentation".
    Source: https://github.com/yyliu01/RPL/blob/main/rpl_corocl.code/loss/PositiveEnergy.py
    """
    n_prob = torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)

    n_entropy = -torch.sum(n_prob * torch.log(n_prob), dim=1) / t
    a_entropy = -torch.sum(a_prob * torch.log(a_prob), dim=1) / t

    entropy_disimilarity = F.mse_loss(
        input=a_entropy, target=n_entropy, reduction="none"
    )
    assert ~torch.isnan(entropy_disimilarity).any()

    return entropy_disimilarity


def energy_loss(logits, targets, vanilla_logits, out_idx, t=1.0):
    """
    Energy-based loss for pixel-wise OOD detection.

    Adapted from:
    Liu et al., "Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection
    in Semantic Segmentation".
    Source: https://github.com/yyliu01/RPL/blob/main/rpl_corocl.code/loss/PositiveEnergy.py
    """
    out_msk = targets == out_idx
    void_msk = targets == 255

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)
    outlier_msk = out_msk | void_msk

    entropy_part = F.cross_entropy(
        input=logits, target=pseudo_targets, reduction="none"
    )[~outlier_msk]

    reg = disimilarity_entropy(logits, vanilla_logits)[~outlier_msk]

    if torch.sum(out_msk) > 0:
        logits_flat = logits.flatten(start_dim=2).permute(0, 2, 1)
        energy_part = F.relu(
            torch.logsumexp(logits_flat, dim=2)[out_msk.flatten(start_dim=1)]
        ).mean()
    else:
        energy_part = torch.tensor(0.0, device=targets.device)

    inlier_loss = entropy_part.mean() + reg.mean()
    outlier_loss = energy_part * 0.05

    return inlier_loss + outlier_loss


def energy_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vanilla_logits: torch.Tensor,
    out_idx: int,
    t: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Energy-Entropy Loss.

    Adapted from:
    Song Xia et al., "Open-set Anomaly Segmentation in Complex Scenarios".
    """

    out_msk = targets == out_idx
    void_msk = targets == 255

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)
    outlier_msk = out_msk | void_msk

    entropy_part = F.cross_entropy(logits, pseudo_targets, reduction="none")[
        ~outlier_msk
    ]

    reg = disimilarity_entropy(logits, vanilla_logits)[~outlier_msk]

    if torch.sum(out_msk) > 0:

        eps = 1e-6
        prob = (
            torch.clamp(torch.softmax(logits, dim=1), eps)
            .flatten(start_dim=2)
            .permute(0, 2, 1)
        )
        logits_flat = logits.flatten(start_dim=2).permute(0, 2, 1)

        energy = torch.logsumexp(logits_flat, dim=2)
        entropy = -torch.sum(prob * torch.log(prob), dim=2)

        sigmoid_energy = torch.clamp(torch.sigmoid(-energy), eps, 1 - eps)

        outlier_flat = out_msk.flatten(start_dim=1)
        inlier_flat = ~outlier_flat

        outlier_part = (
            -torch.log(sigmoid_energy[outlier_flat]) - alpha * entropy[outlier_flat]
        )
        inlier_part = (
            torch.log(1 - sigmoid_energy[inlier_flat]) + alpha * entropy[inlier_flat]
        )

        energy_entropy = outlier_part.mean() - inlier_part.mean()

    else:
        energy_entropy = torch.tensor(0.0, device=targets.device)

    inlier_loss = entropy_part.mean() + reg.mean()
    outlier_loss = energy_entropy * 0.05

    return inlier_loss + outlier_loss
