from typing import Tuple

import torch
import torch.nn as nn


class ReLU(nn.ReLU):
    """Apply the rectified linear unit (ReLU) activation function to an array, elementwise."""

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Apply the rectified linear unit (ReLU) activation function to an array, elementwise.

        Args:
            x: An array.

        Returns:
            The sigmoid-activated array.

        """
        return super().forward(*x),
