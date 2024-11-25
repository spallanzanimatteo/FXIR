from typing import Tuple

import torch
import torch.nn as nn


class Conv2d(nn.Conv2d):  # TODO: factor out padding and bias
    """Apply a cross-correlation transformation (AKA convolution) to an input array with two spatial dimensions."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply a cross-correlation transformation (AKA convolution) to an input array with two spatial dimensions.

        Args:
            x: An input array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the cross-correlated array.

        """
        ys = super().forward(x),
        return ys
