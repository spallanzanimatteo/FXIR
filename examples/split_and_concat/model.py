from typing import Tuple

import torch
import torch.nn as nn

import fx_ir.operators as operators


class SplitAndConcatModel(nn.Module):
    """Swap the blocks of an array along a given axis."""

    def __init__(self, dim: int, block_sizes: Tuple[int, int]) -> None:
        """Initialize the model.

        Args:
            dim: The identifier of the axis along which to split and concatenate blocks.
            block_sizes: The number of items that end up in each block along the given axis.

        """
        super().__init__()
        self.register_module("inputs_0", operators.Array(0))
        self.register_module("split", operators.Split(dim, block_sizes))
        self.register_module("split_0", operators.Array(0))
        self.register_module("split_1", operators.Array(1))
        self.register_module("concat", operators.Concat(dim))
        self.register_module("concat_0", operators.Array(0))

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
        """Swap the blocks of an array along a given axis.

        Args:
            inputs: The singleton collection of model inputs.

        Returns:
            The collection of model outputs.

        """
        inputs_0, = self.inputs_0(inputs),  # unpack model inputs
        split = self.split(inputs_0)  # execute operation (pack outputs)
        split_0, split_1 = self.split_0(split), self.split_1(split)  # unpack operation outputs
        concat = self.concat(split_1, split_0)  # execute operation (pack outputs)
        concat_0, = self.concat_0(concat)  # unpack operation outputs
        return concat_0,  # pack model outputs
