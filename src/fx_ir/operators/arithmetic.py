from typing import Tuple

import torch
import torch.nn as nn


class Add(nn.Module):
    """Add two arrays, elementwise."""

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor]:
        """Add two arrays, elementwise.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the elementwise sum of the arrays.

        """
        ys = torch.add(x0, x1),
        return ys


class Mul(nn.Module):
    """Multiply two arrays, elementwise."""

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor]:
        """Multiply two arrays, elementwise.

        Args:
            x0: An array.
            x1: Another array.

        Returns:
            A sequence of arrays. The arrays have the following semantics:
                * the elementwise product of the arrays.

        """
        ys = torch.mul(x0, x1),
        return ys
