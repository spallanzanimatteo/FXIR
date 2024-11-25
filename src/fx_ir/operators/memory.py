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

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor]:
        """Concatenate two arrays along a given axis.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the concatenation of the arrays.

        """
        ys = torch.cat((x0, x1), dim=self.dim),
        return ys


class Flatten(nn.Module):
    """Remove a range of contiguous axes from an array."""

    def __init__(self, dim_start: int, dim_end: int) -> None:
        """Initialize the operation.

        Args:
            dim_start: The identifier of the first axis of the range.
            dim_end: The identifier of the last axis of the range.

        """
        super().__init__()
        self.dim_start = dim_start
        self.dim_end = dim_end

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Remove a range of contiguous axes from an array.

        Args:
            x: An array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the array whose axes have been removed.

        """
        ys = torch.flatten(x, start_dim=self.dim_start, end_dim=self.dim_end),
        return ys


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an array into two blocks along a given axis.

        Args:
            x: An array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the first block of the array along the given axis;
                * the second block of the array along the given axis.

        """
        ys = cast(Tuple[torch.Tensor, torch.Tensor], torch.split(x, dim=self.dim, split_size_or_sections=cast(List[int], self.block_sizes)))
        return ys
