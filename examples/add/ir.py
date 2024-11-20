from typing import Tuple

import torch
import torch.fx as fx
import torch.nn as nn

import fx_ir.operators as operators
import fx_ir.validator as validator


def _create_model() -> fx.GraphModule:
    """Create a model adding an array to itself.

    Returns:
        The model adding an array to itself.

    """
    # define the `Graph`
    graph = fx.Graph()
    inputs = graph.placeholder("inputs")
    inputs_0 = graph.call_module("inputs_0", (inputs,))
    add = graph.call_module("add",(inputs_0, inputs_0))
    add_0 = graph.call_module("add_0", (add,))
    outputs = graph.output((add_0,))
    # define the `Module`
    module = nn.Module()
    module.register_module("inputs_0", operators.Array(0))
    module.register_module("add", operators.Add())
    module.register_module("add_0", operators.Array(0))
    # create the `GraphModule`
    graph_module = fx.GraphModule(root=module, graph=graph)
    return graph_module


def _create_inputs(n: int) -> Tuple[torch.Tensor]:
    """Create the collection of inputs for the addition model.

    Args:
        n: The number of items in the input array.

    Returns:
        The collection of inputs for the addition model.

    """
    return torch.arange(0, n).to(dtype=torch.float32),


def create_model_and_inputs(n: int) -> Tuple[fx.GraphModule, Tuple[torch.Tensor]]:
    """Create a model adding an array to itself, along with the collection of its inputs.

    Args:
        n: The number of items in the input array.

    Returns:
        The model along with the collection of its inputs.

    """
    graph_module = _create_model()
    inputs = _create_inputs(n)
    return graph_module, inputs


def _create_expected_outputs(n: int) -> Tuple[torch.Tensor]:
    """Create the collection of expected model outputs.

    Args:
        n: The number of items in the input array.

    Returns:
        The collection of expected model outputs.

    """
    return 2 * torch.arange(0, n).to(dtype=torch.float32),


def main():
    """Create an FX IR model along with a collection of inputs, validate the model, then apply it to the inputs."""
    n = 2**4
    model, inputs = create_model_and_inputs(n)
    validator.VALIDATOR.validate(model)
    expected_outputs = _create_expected_outputs(n)
    actual_outputs = model(inputs)
    assert all(torch.equal(a, e) for a, e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
