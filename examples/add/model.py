from typing import Tuple

import torch
import torch.nn as nn

import fx_ir.operators as operators


class AddModel(nn.Module):
    """A model adding an array to itself."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.register_module("inputs_0", operators.Array(0))
        self.register_module("add", operators.Add())
        self.register_module("add_0", operators.Array(0))

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
        """Add an array to itself.

        Args:
            inputs: The singleton collection of model inputs.

        Returns:
            The collection of model outputs.

        """
        inputs_0, = self.inputs_0(inputs),  # unpack model inputs
        add = self.add(inputs_0, inputs_0)  # execute operation (pack outputs)
        add_0, = self.add_0(add),  # unpack operation outputs
        return add_0,  # pack model outputs
