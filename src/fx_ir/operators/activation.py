from typing import Tuple

import torch
import torch.nn as nn


class ReLU(nn.ReLU):
    """Apply the rectified linear unit (ReLU) activation function to an array, elementwise."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply the rectified linear unit (ReLU) activation function to an array, elementwise.

        Args:
            x: An array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the ReLU-activated array.

        """
        return super().forward(x),
