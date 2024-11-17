from typing import Tuple

import torch
import torch.fx as fx
import torch.nn as nn

import fx_ir.operators as operators
import fx_ir.validator as validator


def _create_model(n_input: int, n_hidden: int, n_output: int) -> fx.GraphModule:
    """Create the model of a perceptron.

    Args:
        n_input: The number of input units.
        n_hidden: The number of hidden units.
        n_output: The number of output units.

    Returns:
        The model of a perceptron.

    """
    # define the `Graph`
    graph = fx.Graph()
    inputs = graph.placeholder("inputs")
    inputs_0 = graph.call_module("inputs_0", (inputs,))
    linear1 = graph.call_module("linear1",(inputs_0,))
    linear1_0 = graph.call_module("linear1_0", (linear1,))
    relu1 = graph.call_module("relu1", (linear1_0,))
    relu1_0 = graph.call_module("relu1_0", (relu1,))
    linear2 = graph.call_module("linear2", (relu1_0,))
    linear2_0 = graph.call_module("linear2_0", (linear2,))
    outputs = graph.output((linear2_0,))
    # define the `Module`
    module = nn.Module()
    module.register_module("inputs_0", operators.Array(0))
    module.register_module("linear1", operators.Linear(n_input, n_hidden))
    module.register_module("linear1_0", operators.Array(0))
    module.register_module("relu1", operators.ReLU())
    module.register_module("relu1_0", operators.Array(0))
    module.register_module("linear2", operators.Linear(n_hidden, n_output))
    module.register_module("linear2_0", operators.Array(0))
    # set weights and biases to one
    with torch.no_grad():
        for m in module.children():
            if isinstance(m, operators.Linear):
                m.weight.fill_(1)
                m.bias.fill_(1)
    # create the `GraphModule`
    graph_module = fx.GraphModule(root=module, graph=graph)
    return graph_module


def _create_inputs(batch_size: int, n_input: int) -> Tuple[torch.Tensor]:
    """Create the collection of perceptron inputs.

    Args:
        batch_size: The number of data points in the batch.
        n_input: The size of a single data point.

    Returns:
        The collection of perceptron inputs.

    """
    return torch.ones(batch_size, n_input),


def create_model_and_inputs(batch_size: int, n_input: int, n_hidden: int, n_output: int) -> Tuple[fx.GraphModule, Tuple[torch.Tensor]]:
    """Create a perceptron model, along with the collection of its inputs.

    Args:
        batch_size: The number of data points in the batch.
        n_input: The number of input units.
        n_hidden: The number of hidden units.
        n_output: The number of output units.

    Returns:
        The model along with the collection of its inputs.

    """
    graph_module = _create_model(n_input, n_hidden, n_output)
    inputs = _create_inputs(batch_size, n_input)
    return graph_module, inputs


def _create_expected_outputs(batch_size: int, n_input: int, n_hidden: int, n_output: int) -> Tuple[torch.Tensor]:
    """Create the collection of expected model outputs.

    Args:
        batch_size: The number of data points in the batch.
        n_input: The number of input units.
        n_hidden: The number of hidden units.
        n_output: The number of output units.

    Returns:
        The collection of expected model outputs.

    """
    return torch.full((batch_size, n_output), ((n_input + 1) * n_hidden) + 1),


def main():
    """Create an FX IR model with a collection of inputs, validate the model, then apply it to the inputs."""
    batch_size, n_input, n_hidden, n_output = 2**4, 2**2, 2**3, 2**1
    model, inputs = create_model_and_inputs(batch_size, n_input, n_hidden, n_output)
    validator.validate(model)
    actual_outputs = model(inputs)
    expected_outputs = _create_expected_outputs(batch_size, n_input, n_hidden, n_output)
    assert all(torch.equal(a, e) for (a,), e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
