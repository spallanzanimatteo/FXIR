from typing import Tuple

import torch
import torch.nn as nn


class Array(nn.Module):
    """Extract an array from an array sequence."""

    def __init__(self, index: int) -> None:
        """Initialize the operation.

        Args:
            index: The position of the array in the sequence.

        """
        super().__init__()
        self.index = index

    def forward(self, ys: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extract an array from an array sequence.

        Args:
            ys: An array sequence.

        Returns:
            An array.

        """
        x = ys[self.index]
        return x
