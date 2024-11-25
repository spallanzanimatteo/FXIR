from typing import Tuple

import torch
import torch.fx as fx

import fx_ir.editor as editor
import fx_ir.operators as operators


def _create_model() -> fx.GraphModule:
    """Create a model adding an array to itself.

    Returns:
        The model adding an array to itself.

    """
    editor_ = editor.Editor(debug=False)
    i0 = editor_.add_input()
    a, (a0,) = editor_.add_operation(editor_.output, "add", operators.Add(), {(0, None): i0, (1, None): i0}, {})
    editor_.add_output(a0)
    graph_module = editor_.finalize()
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
    expected_outputs = _create_expected_outputs(n)
    actual_outputs = model(inputs)
    assert all(torch.equal(a, e) for a, e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
