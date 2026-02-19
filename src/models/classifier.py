"""EfficientNet-B0 classifier with configurable output head.

Wraps timm's EfficientNet-B0 (timm 1.0.15) in a thin nn.Module that:
- Exposes raw logits (no softmax) for CrossEntropyLoss compatibility
- Reports num_features (1280) for downstream feature extraction
- Provides gradcam_target_layer property for Phase 6 Grad-CAM

Architecture (EfficientNet-B0 via timm):
- Backbone: EfficientNet-B0 with ImageNet-pretrained weights
- Classifier head: Linear(1280, num_classes) -- replaced by timm
- Dropout: Configurable via drop_rate (default 0.2)
- Total parameters: 4,011,391 (with num_classes=3)
- Full fine-tuning: All parameters trainable (no frozen layers)

Implemented in Phase 4.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class BTXRDClassifier(nn.Module):
    """EfficientNet-B0 classifier for 3-class bone tumor classification.

    Produces raw logits (NOT softmax) -- use with nn.CrossEntropyLoss.

    Args:
        backbone: timm model name (default: "efficientnet_b0").
        num_classes: Number of output classes (default: 3).
        pretrained: Whether to load ImageNet-pretrained weights.
        drop_rate: Dropout rate applied before classifier head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        self.num_classes = num_classes
        self.num_features = self.model.num_features  # 1280 for efficientnet_b0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.model(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the layer to use for Grad-CAM visualization.

        Returns the final BatchNormAct2d (1280 channels) before global
        average pooling -- confirmed as the correct Grad-CAM target for
        EfficientNet-B0 with pytorch-grad-cam 1.5.5.
        """
        return self.model.bn2
