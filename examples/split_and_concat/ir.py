from typing import Tuple

import torch
import torch.fx as fx

import fx_ir.editor as editor
import fx_ir.operators as operators


def _create_model(block_sizes: Tuple[int, int]) -> fx.GraphModule:
    """Create a model swapping two blocks of an array along a given axis.

    Args:
        block_sizes: The number of items that end up in each block along the given axis.

    Returns:
        The model swapping two blocks of an array.

    """
    editor_ = editor.Editor(debug=False)
    i0 = editor_.add_input()
    s, (s0, s1) = editor_.add_operation(editor_.output, "split", operators.Split(0, block_sizes), {(0, None): i0}, {})
    c, (c0,) = editor_.add_operation(editor_.output, "concat", operators.Concat(0), {(0, None): s1, (1, None): s0}, {})
    editor_.add_output(c0)
    graph_module = editor_.finalize()
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
    actual_outputs = model(inputs)
    expected_outputs = _create_expected_outputs(block_sizes)
    assert all(torch.equal(a, e) for a, e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
