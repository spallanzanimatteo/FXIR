from typing import Tuple

import torch
import torch.nn as nn


class BatchNorm2d(nn.BatchNorm2d):
    """Apply a batch-normalization transformation to an input array with two spatial dimensions."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply a batch-normalization transformation to an input array with two spatial dimensions.

        Args:
            x: An input array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the affine-transformed array.

        """
        ys = super().forward(x),
        return ys
