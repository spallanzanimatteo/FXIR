from typing import Tuple

import torch
import torch.fx as fx

import fx_ir.editor as editor
import fx_ir.operators as operators


def _create_model(n_input: int, n_hidden: int, n_output: int) -> fx.GraphModule:
    """Create the model of a perceptron.

    Args:
        n_input: The number of input units.
        n_hidden: The number of hidden units.
        n_output: The number of output units.

    Returns:
        The model of a perceptron.

    """
    # create the model
    editor_ = editor.Editor(debug=False)
    i0 = editor_.add_input()
    l1, (l1_0,) = editor_.add_operation(editor_.output, "linear1", operators.Linear(n_input, n_hidden), {(0, None): i0}, {})
    r1, (r1_0,) = editor_.add_operation(editor_.output, "relu1", operators.ReLU(), {(0, None): l1_0}, {})
    l2, (l2_0,) = editor_.add_operation(editor_.output, "linear2", operators.Linear(n_hidden, n_output), {(0, None): r1_0}, {})
    editor_.add_output(l2_0)
    graph_module = editor_.finalize()
    # set weights and biases to one
    with torch.no_grad():
        for module in graph_module.children():
            if isinstance(module, operators.Linear):
                module.weight.fill_(1)
                module.bias.fill_(1)
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
    actual_outputs = model(inputs)
    expected_outputs = _create_expected_outputs(batch_size, n_input, n_hidden, n_output)
    assert all(torch.equal(a, e) for a, e in zip(actual_outputs, expected_outputs))


if __name__ == "__main__":
    main()
