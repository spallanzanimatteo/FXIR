from typing import Tuple

import torch
import torch.nn as nn

import fx_ir.operators as operators


class PerceptronModel(nn.Module):
    """A perceptron."""

    def __init__(self, n_input: int, n_hidden: int, n_output: int) -> None:
        """Initialize the model.

        Args:
            n_input: The number of input units.
            n_hidden: The number of hidden units.
            n_output: The number of output units.

        """
        super().__init__()
        self.register_module("inputs_0", operators.Array(0))
        self.register_module("linear1", operators.Linear(n_input, n_hidden))
        self.register_module("linear1_0", operators.Array(0))
        self.register_module("relu1", operators.ReLU())
        self.register_module("relu1_0", operators.Array(0))
        self.register_module("linear2", operators.Linear(n_hidden, n_output))
        self.register_module("linear2_0", operators.Array(0))
        self._initialize()

    def _initialize(self) -> None:
        """Set the perceptron's weights and biases to one."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, operators.Linear):
                    m.weight.fill_(1)
                    m.bias.fill_(1)

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
        """Apply the perceptron to a batch of input data points.

        Args:
            inputs: The collection of model inputs.

        Returns:
            The collection of model outputs.

        """
        inputs_0, = self.inputs_0(inputs),  # unpack model inputs
        linear1 = self.linear1(inputs_0)  # execute operation (pack outputs)
        linear1_0, =  self.linear1_0(linear1),  # unpack operation outputs
        relu1 = self.relu1(linear1_0)  # execute operation (pack outputs)
        relu1_0, = self.relu1_0(relu1),  # unpack operation outputs
        linear2 = self.linear2(relu1_0)  # execute operation (pack outputs)
        linear2_0, = self.linear2_0(linear2),  # unpack operation outputs
        return linear2_0,  # pack model outputs
