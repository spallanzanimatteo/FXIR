from typing import Tuple

import torch
import torch.nn as nn


class Add(nn.Module):
    """Add two arrays, elementwise."""

    def forward(self, x0: Tuple[torch.Tensor], x1: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Add two arrays, elementwise.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            The elementwise sum of the arrays.

        """
        return torch.add(*x0, *x1),


class Mul(nn.Module):
    """Multiply two arrays, elementwise."""

    def forward(self, x0: Tuple[torch.Tensor], x1: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Multiply two arrays, elementwise.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            The elementwise product of the arrays.

        """
        return torch.mul(*x0, *x1),
