from typing import Tuple

import torch
import torch.nn as nn


class Linear(nn.Linear):
    """Apply an affine transformation to an input array."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply an affine transformation to an input array.

        Args:
            x: An array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the affine-transformed array.

        """
        ys = super().forward(x),
        return ys
