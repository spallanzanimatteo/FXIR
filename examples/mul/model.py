from typing import Tuple

import torch
import torch.nn as nn

import fx_ir.operators as operators


class MulModel(nn.Module):
    """A model multiplying an array by itself."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.register_module("inputs_0", operators.Array(0))
        self.register_module("mul", operators.Mul())
        self.register_module("mul_0", operators.Array(0))

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
        """Multiply an array by itself.

        Args:
            inputs: The singleton collection of model inputs.

        Returns:
            The collection of model outputs.

        """
        inputs_0, = self.inputs_0(inputs),  # unpack model inputs
        mul = self.mul(inputs_0, inputs_0)  # execute operation (pack outputs)
        mul_0, = self.mul_0(mul),  # unpack operation outputs
        return mul_0,  # pack model outputs
