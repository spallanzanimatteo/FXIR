from typing import Tuple

import torch
import torch.nn as nn


class Linear(nn.Linear):
    """Apply an affine transformation to an input array."""

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Apply an affine transformation to an input array.

        Args:
            x: An array.

        Returns:
            The affine-transformed array.

        """
        return super().forward(*x),
