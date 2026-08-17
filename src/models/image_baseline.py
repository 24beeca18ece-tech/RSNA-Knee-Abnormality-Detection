"""v0 unimodal image baseline model. See docs/baseline_plan.md for rationale.

Single 2D slice in, 12 sigmoid logits out (one per target, multi-label).
Intentionally simple: swap the backbone or add a text branch later rather
than growing this file — see docs/baseline_plan.md "After v0" section.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def build_backbone(name: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Shared backbone builder - used here for the supervised classifier
    head and by src/models/ssl_model.py for self-supervised pretraining
    (Step 3), so a backbone pretrained one way loads cleanly into the other
    (same module structure, just a different head on top)."""
    weights = "IMAGENET1K_V1" if pretrained else None
    if name == "resnet18":
        net = torchvision.models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        net = torchvision.models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return net, in_features


class KneeImageBaseline(nn.Module):
    def __init__(self, n_targets: int = 12, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        net, in_features = build_backbone(backbone, pretrained)
        self.backbone = net
        self.head = nn.Linear(in_features, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)  # raw logits; use BCEWithLogitsLoss


if __name__ == "__main__":
    # Smoke test: no data/weights download needed beyond torchvision's
    # pretrained resnet18 (skip that part offline by passing pretrained=False).
    model = KneeImageBaseline(n_targets=12, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 12), out.shape
    print(f"OK: output shape {tuple(out.shape)}")
