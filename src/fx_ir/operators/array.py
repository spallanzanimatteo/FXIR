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

    def forward(self, collection: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extract an array from an array sequence.

        Args:
            collection: An array sequence.

        Returns:
            An array.

        """
        return collection[self.index]
