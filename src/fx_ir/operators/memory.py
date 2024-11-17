from typing import List, Tuple, cast

import torch
import torch.nn as nn


class Concat(nn.Module):
    """Concatenate two arrays along a given axis."""

    def __init__(self, dim: int) -> None:
        """Initialize the operation.

        Args:
            dim: The identifier of the axis along which the arrays are concatenated.

        """
        super().__init__()
        self.dim = dim

    def forward(self, x0: Tuple[torch.Tensor], x1: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Concatenate two arrays along a given axis.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            The concatenation of the arrays.

        """
        return torch.cat((*x0, *x1), dim=self.dim),


class Split(nn.Module):
    """Split an array into two blocks along a given axis."""

    def __init__(self, dim: int, block_sizes: Tuple[int, int]) -> None:
        """Initialize the operation.

        Args:
            dim: The axis along which the array is split.
            block_sizes: The number of items that end up in each block along the given axis.

        """
        super().__init__()
        self.dim = dim
        self.block_sizes = block_sizes

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an array into two blocks along a given axis.

        Args:
            x: An array.

        Returns:
            The two blocks.

        """
        return cast(Tuple[torch.Tensor, torch.Tensor], torch.split(*x, dim=self.dim, split_size_or_sections=cast(List[int], self.block_sizes)))
