import torch
import torch.nn.functional as F


def disimilarity_entropy(logits, vanilla_logits, t=1.0):
    """
    Based on:
    https://github.com/yyliu01/RPL/blob/main/rpl_corocl.code/loss/PositiveEnergy.py
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
    Based on:
    https://github.com/yyliu01/RPL/blob/main/rpl_corocl.code/loss/PositiveEnergy.py
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
