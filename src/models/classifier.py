"""Multi-architecture classifier with configurable backbone and output head.

Wraps timm model families in a thin nn.Module that:
- Exposes raw logits (no softmax) for CrossEntropyLoss compatibility
- Reports num_features for downstream feature extraction
- Provides gradcam_target_layer property for Grad-CAM visualization
- Passes **kwargs through to timm.create_model (e.g., block_args for CBAM)

Supports EfficientNet-B0 through B7, ResNet-50 (with optional CBAM attention),
and other timm backbones.

Implemented in Phase 4, updated in Phase 9 for multi-architecture support.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class BTXRDClassifier(nn.Module):
    """Multi-architecture classifier for 3-class bone tumor classification.

    Produces raw logits (NOT softmax) -- use with nn.CrossEntropyLoss.

    Supported architectures:
    - EfficientNet-B0 through B7 (gradcam via model.bn2)
    - ResNet-50 with optional CBAM attention (gradcam via model.layer4[-1])

    Args:
        backbone: timm model name (e.g., "efficientnet_b0", "resnet50").
        num_classes: Number of output classes (default: 3).
        pretrained: Whether to load ImageNet-pretrained weights.
        drop_rate: Dropout rate applied before classifier head.
        **kwargs: Additional arguments passed to timm.create_model
            (e.g., block_args=dict(attn_layer='cbam') for CBAM attention).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            **kwargs,
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

        Architecture-specific target layers:
        - EfficientNet: model.bn2 (final BatchNormAct2d before GAP)
        - ResNet: model.layer4[-1] (last bottleneck block)

        Falls back to the last convolutional module if neither is present.
        """
        if hasattr(self.model, "bn2"):
            return self.model.bn2  # EfficientNet family
        if hasattr(self.model, "layer4"):
            return self.model.layer4[-1]  # ResNet family
        # Fallback: walk backwards through modules to find last BatchNorm or Conv2d
        for module in reversed(list(self.model.modules())):
            if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
                return module
        raise AttributeError(
            f"Cannot find Grad-CAM target layer for backbone. "
            f"Model has no bn2 or layer4 attribute and no "
            f"BatchNorm2d/Conv2d modules."
        )
