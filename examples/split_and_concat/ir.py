from typing import Tuple

import torch
import torch.fx as fx
import torch.nn as nn

import fx_ir.operators as operators
import fx_ir.validator as validator


def _create_model(block_sizes: Tuple[int, int]) -> fx.GraphModule:
    """Create a model swapping two blocks of an array along a given axis.

    Args:
        block_sizes: The number of items that end up in each block along the given axis.

    Returns:
        The model swapping two blocks of an array.

    """
    # define the `Graph`
    graph = fx.Graph()
    inputs = graph.placeholder("inputs")
    inputs_0 = graph.call_module("inputs_0", (inputs,))
    split = graph.call_module("split",(inputs_0,))
    split_0 = graph.call_module("split_0", (split,))
    split_1 = graph.call_module("split_1", (split,))
    concat = graph.call_module("concat", (split_1, split_0))
    concat_0 = graph.call_module("concat_0", (concat,))
    outputs = graph.output((concat_0,))
    # define the `Module`
    module = nn.Module()
    module.register_module("inputs_0", operators.Array(0))
    module.register_module("split", operators.Split(0, block_sizes))
    module.register_module("split_0", operators.Array(0))
    module.register_module("split_1", operators.Array(1))
    module.register_module("concat", operators.Concat(0))
    module.register_module("concat_0", operators.Array(0))
    # create the `GraphModule`
    graph_module = fx.GraphModule(root=module, graph=graph)
    return graph_module


def _create_inputs(block_sizes: Tuple[int, int]) -> Tuple[torch.Tensor]:
    """Create the collection of inputs for the block swapping model.

    Args:
        block_sizes: The number of items that end up in each block along the given axis.

    Returns:
        The collection of inputs for the block swapping model.

    """
    n0, n1 = block_sizes
    return torch.vstack([torch.full((n0, 1), 0), torch.full((n1, 1), 1)]),


def create_model_and_inputs(block_sizes: Tuple[int, int]) -> Tuple[fx.GraphModule, Tuple[torch.Tensor]]:
    """Create a model swapping two blocks of an array, along with the collection of its inputs.

    Args:
        block_sizes: The number of items that end up in each block along the given axis.

    Returns:
        The model along with the collection of its inputs.

    """
    graph_module = _create_model(block_sizes)
    inputs = _create_inputs(block_sizes)
    return graph_module, inputs


def _create_expected_outputs(block_sizes: Tuple[int, int]) -> Tuple[torch.Tensor]:
    """Create the collection of expected model outputs.

    Args:
        block_sizes: The number of items that end up in each block along the given axis.

    Returns:
        The collection of expected model outputs.

    """
    n0, n1 = block_sizes
    return torch.vstack([torch.full((n1, 1), 1), torch.full((n0, 1), 0)]),


def main():
    """Create an FX IR model with a collection of inputs, validate the model, then apply it to the inputs."""
    block_sizes = (2**3, 2**4)
    model, inputs = create_model_and_inputs(block_sizes)
    validator.validate(model)
    actual_outputs = model(inputs)
    expected_outputs = _create_expected_outputs(block_sizes)
    assert all(torch.equal(a, e) for a, e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
