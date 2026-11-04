"""
MobileNetV3-Small architecture adapted for CIFAR-10.
"""

from typing import Any
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def mobilenet_v3_small_cifar(num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """
    Constructs a MobileNetV3-Small model adapted for 32x32 CIFAR-10 inputs.

    Standard ImageNet MobileNetV3-Small downsamples 224x224 inputs by a total factor
    of 32 through 5 stride-2 stages (stem at features[0], IR 0 at features[1],
    IR 1 at features[2], IR 3 at features[4], and IR 8 at features[9]).
    On 32x32 images, a 32x reduction collapses spatial dimensions to 1x1 at IR 8,
    which causes BatchNorm2d in training mode to fail when batch size is 1
    (as N * H * W = 1 * 1 * 1 = 1, violating the requirement of >1 element to compute
    batch variance).

    This CIFAR-10 adaptation:
      1. Changes stem convolution stride from 2 to 1 (features[0][0].stride = (1, 1)),
         yielding a 16x total spatial reduction across remaining stages:
           - features[0] (stem): 32x32 (16 ch)
           - features[1] (IR 0, s=2): 16x16 (16 ch)
           - features[2] (IR 1, s=2): 8x8 (24 ch)
           - features[3] (IR 2, s=1): 8x8 (24 ch)
           - features[4] (IR 3, s=2): 4x4 (40 ch)
           - features[5-8] (IR 4-7, s=1): 4x4 (40-48 ch)
           - features[9] (IR 8, s=2): 2x2 (96 ch)
           - features[10-11] (IR 9-10, s=1): 2x2 (96 ch)
           - features[12] (Pointwise, s=1): 2x2 (576 ch)
           - avgpool: 1x1 (576 ch)
           - classifier[0] (Linear): 1024 ch
           - classifier[3] (Linear): num_classes logits
      2. Replaces the final classifier linear projection (classifier[3]) to
         output num_classes logits.

    Parameters:
        num_classes: Number of output classification logits (default: 10).
        **kwargs: Optional keyword arguments passed to mobilenet_v3_small.

    Returns:
        Adapted MobileNetV3-Small model instance (torch.nn.Module).
    """
    model = mobilenet_v3_small(weights=None, **kwargs)

    # Stem adaptation: stride 1 instead of stride 2 to preserve spatial resolution
    model.features[0][0].stride = (1, 1)

    # Classification head for CIFAR-10 classes
    model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)

    return model
