"""SimCLR-style contrastive model for Step 3 self-supervised pretraining.
See docs/baseline_plan.md "Step 3" for why: pretrain the image encoder on
all 4407 studies' images (no labels needed) before fine-tuning a classifier
head on the 813-row combined labeled set from Step 2, so the still-3594
unlabeled studies' *images* aren't left completely unused.

Backbone matches src/models/image_baseline.py (build_backbone) exactly, so
a backbone pretrained here loads directly into KneeImageBaseline for
fine-tuning - see src/training/finetune.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.image_baseline import build_backbone


class SimCLRModel(nn.Module):
    """backbone -> projection head (2-layer MLP, standard SimCLR). The
    projection head is only used to compute the contrastive loss during
    pretraining and is discarded afterward - only `backbone`'s weights get
    carried into fine-tuning."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, projection_dim: int = 128):
        super().__init__()
        net, in_features = build_backbone(backbone, pretrained)
        self.backbone = net
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        proj = self.projection_head(features)
        return F.normalize(proj, dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (SimCLR's contrastive
    loss). z1, z2: (B, D) L2-normalized projections of the two augmented
    views of the same B images - positives are (z1[i], z2[i]), negatives
    are every other sample in the combined 2B-item batch."""
    batch_size = z1.size(0)
    device = z1.device
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, float("-inf"))

    positive_idx = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device),
    ])
    return F.cross_entropy(sim, positive_idx)


if __name__ == "__main__":
    model = SimCLRModel(pretrained=False)
    v1, v2 = torch.randn(4, 3, 224, 224), torch.randn(4, 3, 224, 224)
    z1, z2 = model(v1), model(v2)
    loss = nt_xent_loss(z1, z2)
    print(f"OK: z shape {tuple(z1.shape)}, loss {loss.item():.4f}")
