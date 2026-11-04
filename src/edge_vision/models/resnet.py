"""
ResNet18 architecture adapted for CIFAR-10.
"""

from typing import Any
import torch.nn as nn
from torchvision.models import resnet18


def resnet18_cifar(num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """
    Constructs a ResNet-18 model adapted for 32x32 CIFAR-10 inputs.

    Standard ImageNet ResNet-18 applies a 7x7 conv with stride 2 and a 3x3 maxpool
    with stride 2 in the stem, downsampling 224x224 inputs by a factor of 4 to 56x56
    before entering layer1. When applied to 32x32 CIFAR-10 images, this stem would
    prematurely reduce spatial dimensions to 8x8 before residual processing.

    This CIFAR-10 adaptation:
      1. Replaces conv1 (7x7, stride 2, pad 3) with conv1 (3x3, stride 1, pad 1, no bias),
         preserving the initial 32x32 spatial resolution.
      2. Replaces the maxpool (3x3, stride 2, pad 1) with nn.Identity(), avoiding
         premature downsampling.
      3. Replaces the final fully connected layer (fc) to output num_classes logits.

    The four residual stages downsample via stride 2 convolutions as follows:
      - Stem: 32x32 -> 32x32 (64 channels)
      - layer1: 32x32 (64 channels)
      - layer2: 16x16 (128 channels)
      - layer3: 8x8 (256 channels)
      - layer4: 4x4 (512 channels)
      - avgpool: 1x1 (512 channels)
      - fc: num_classes logits

    Parameters:
        num_classes: Number of output classification logits (default: 10).
        **kwargs: Optional keyword arguments passed to resnet18.

    Returns:
        Adapted ResNet-18 model instance (torch.nn.Module).
    """
    model = resnet18(weights=None, **kwargs)

    # Stem adaptation: 3x3 conv with stride 1, padding 1, no bias
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove downsampling maxpool
    model.maxpool = nn.Identity()

    # Classification head for CIFAR-10 classes
    model.fc = nn.Linear(in_features=512, out_features=num_classes)

    return model
