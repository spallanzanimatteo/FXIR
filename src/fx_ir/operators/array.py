from typing import Tuple

import torch
import torch.nn as nn


class Array(nn.Module):
    """An operator producing a singleton array collection."""

    def __init__(self, index: int) -> None:
        """Initialize the operation.

        Args:
            index: The position of the array in the collection.

        """
        super().__init__()
        self.index = index

    def forward(self, collection: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
        """Extract a singleton from an array collection.

        Args:
            collection: A collection of arrays.

        Returns:
            A singleton array collection.

        """
        return collection[self.index],
