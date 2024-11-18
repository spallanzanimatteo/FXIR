import collections
from typing import Dict, List, Tuple, Type, get_args
import warnings

import torch.fx as fx
import torch.nn as nn

import fx_ir.editor.arity_and_co_arity as arity_and_co_arity
import fx_ir.operators as operators
import fx_ir.validator as validator


def _create_empty_graph_module() -> fx.GraphModule:
    """Create an empty FX-IR-compliant model.

    Returns:
        An empty FX-IR-compliant model.

    """
    # create the `Graph`
    graph = fx.Graph()
    _ = graph.placeholder("inputs")
    _ = graph.output(())
    # create the `Module`
    module = nn.Module()
    # create the `GraphModule`
    graph_module = fx.GraphModule(root=module, graph=graph)
    return graph_module


_Array = fx.Node
"""A 'call_module' node representing an array."""

_Operation = fx.Node
"""A 'call_module' node representing an operation."""

_InputArgument = Tuple[int, _Operation]
"""An input argument to an operation node. Can be resolved to a 'call_module' node representing an array."""

_OutputValue = Tuple[_Operation, int]
"""An output value of an operation node. Can be resolved to a 'call_module' node representing an array."""

_InputRef = Tuple[int, None]
"""The input argument of an operation node that has not yet been created."""

_OutputRef = Tuple[None, int]
"""The output value of an operation node that has not yet been created."""


class Editor:
    """Create and edit an FX-IR-compliant model."""

    def __init__(self, debug: bool) -> None:
        """Initialize the editor.

        Args:
            debug: Whether to perform exhaustive sanity checks during editing.

        Attributes:
            graph_module: The model being edited.
            inputs: The arrays that are inputs to the model.
            outputs: The arrays that are outputs of the model.
            reads: For each array, how many times it is read.
            operations: The operations of the model.

        """
        self.debug = debug
        self.graph_module = _create_empty_graph_module()
        self.inputs = dict()
        self.outputs = dict()
        self.reads = collections.Counter()
        self.operations = dict()

    @property
    def orphaned(self) -> List[_Array]:
        """Return the arrays that are neither outputs nor read by any operation.

        Returns:
            The arrays that are neither read by any operation, nor pure outputs of the model. They are dead code.

        """
        orphaned = [node for node, count in self.reads.items() if ((count == 0) and (not (node in self.outputs)))]
        return orphaned

    def toggle(self) -> None:
        """Switch from debug to production mode, and viceversa."""
        self.debug = not self.debug

    def reset(self) -> None:
        """Reset the model being edited."""
        self.graph_module = _create_empty_graph_module()
        self.inputs = dict()
        self.outputs = dict()
        self.reads = collections.Counter()
        self.operations = dict()

    def _check_target(self, target: str) -> None:
        """Check whether the target can be used in the model.

        Args:
            target: The label for the array or operation.

        Raises:
            ValueError: The target is already used.

        """
        if self.graph_module.graph.find_nodes(op="call_module", target=target):
            raise ValueError(f"Target {target} is already used")

    def add_output(self, array: _Array) -> None:
        """Add an output to the model.

        Args:
            array: The new output of the model.

        Warns:
            UserWarning: The node is not an array.

        """
        if array in self.outputs:
            return

        # validate the candidate output
        if not isinstance(self.graph_module.get_submodule(array.target), operators.Array):
            warnings.warn(f"Node {array.target} is not an array", UserWarning)
            return
        # update the model output
        output, = self.graph_module.graph.find_nodes(op="output")
        output.update_arg(0, output.args[0] + (array,))
        # update the bookkeeping data structures
        self.outputs[array] = None

    def remove_output(self) -> None:
        """Remove the last output from the model.

        Warns:
            UserWarning: The array is orphaned (debug-mode-only).

        """
        if len(self.outputs) == 0:
            return

        # compute the output to remove
        output, = self.graph_module.graph.find_nodes(op="output")
        node = output.args[0][-1]
        # update the model
        output.update_arg(0, output.args[0][0:-1])
        # update the bookkeeping data structures
        del self.outputs[node]

        if self.debug and self.reads[node] == 0:
            warnings.warn(f"Array {node.target} is orphaned", UserWarning)

    def add_input(self) -> _Array:
        """Add an input to the model.

        Returns:
            The new input to the model.

        Warns:
            UserWarning: The array is orphaned (debug-mode-only).

        """
        # compute the input to add
        input_, = self.graph_module.graph.find_nodes(op="placeholder")
        inputs, input_idx = list(input_.users), len(input_.users)
        # create the label for the input to add
        target = f"{input_.name}_{input_idx}"
        self._check_target(target)
        # update the model
        with self.graph_module.graph.inserting_after(inputs[-1] if inputs else input_):
            self.graph_module.add_submodule(target, operators.Array(input_idx))
            array = self.graph_module.graph.call_module(target, (input_,))
        # update the bookkeeping data structures
        self.inputs[array] = None
        self.reads[array] = 0
        if self.debug:
            warnings.warn(f"Array {array.target} is orphaned", UserWarning)
        return array

    def remove_input(self) -> None:
        """Remove the last input from the model.

        Warns:
            UserWarning: The input has uses.

        """
        # compute the input to remove
        input_, = self.graph_module.graph.find_nodes(op="placeholder")
        array = list(input_.users)[-1]
        # validate the input to remove
        if self.reads[array] > 0:
            consumers = list(array.users)
            warnings.warn(f"Array {array.target} has uses: {consumers}", UserWarning)
            return
        # update the model
        self.graph_module.graph.erase_node(array)
        self.graph_module.delete_submodule(array.target)
        # update the bookkeeping data structures
        del self.inputs[array]
        del self.reads[array]

    def _validate_add_operation_inputs(
        self, target: str, operation: operators.OperatorT, inputs: Dict[_InputRef, _Array]
    ) -> None:
        """Check whether the `inputs` argument to an `add_operation` call is valid.

        Args:
            target: The label for the operation.
            operation: The FX IR operation to be added to the model.
            inputs: For each input of the new operation, the array that it reads.

        Raises:
            ValueError: The `inputs` argument does not specify all the inputs for the operation.
            ValueError: The `inputs` argument specifies inputs that are not required by the operation.

        """
        arity, _ = arity_and_co_arity.ARITY_AND_CO_ARITY[type(operation)]
        actual, expected = {k for k, _ in inputs}, {i for i in range(arity)}
        underspecified, overspecified = expected - actual, actual - expected
        if underspecified:
            raise ValueError(f"Inputs {[(i, None) for i in sorted(underspecified)]} to operator {target} are not specified")
        if overspecified:
            raise ValueError(f"Inputs {[(i, None) for i in sorted(overspecified)]} to operator {target} are not required")

    def _validate_add_operation_replace(
        self, target: str, operation: operators.OperatorT, replace: Dict[_OutputRef, List[_InputArgument]]
    ) -> None:
        """Check whether the `replace` argument to an `add_operation` call is valid.

        Args:
            target: The label for the operation.
            operation: The FX IR operation to be added to the model.
            replace: For each output of the new operation, specify which downstream operation inputs to replace.

        Warns:
            UserWarning: An array will be orphaned by the transformation.

        Raises:
            ValueError: The `replace` argument does not specify the uses of an operation outputs.
            ValueError: The `replace` argument specifies uses of an array that is not an operation output.
            ValueError: The `replace` argument specifies more than one way to replace the input of a downstream operation.

        """
        _, co_arity = arity_and_co_arity.ARITY_AND_CO_ARITY[type(operation)]
        actual, expected = {k for _, k in replace}, {j for j in range(co_arity)}
        underspecified, overspecified = expected - actual, actual - expected
        if underspecified:
            raise ValueError(f"Outputs {[(None, j) for j in sorted(underspecified)]} of operation {target} are not specified")
        if overspecified:
            raise ValueError(f"Outputs {[(None, j) for j in sorted(overspecified)]} of operation {target} are not required")

        input_argument_to_output_refs = collections.defaultdict(list)
        for output_ref, input_arguments in replace.items():
            for arg in input_arguments:
                input_argument_to_output_refs[arg].append([output_ref])
        for (input_idx, cons), output_refs in input_argument_to_output_refs.items():
            array = cons.args[input_idx]
            if (self.reads[array] == 1) and (not (array in self.outputs)):
                warnings.warn(f"Array {array.target} will be orphaned", UserWarning)
            if len(output_refs) > 1:
                raise ValueError(f"Replacement of {input_idx}-th input of operation {cons.target} is ambiguous")

    def add_operation(
        self,
        target: str,
        operation: operators.OperatorT,
        inputs: Dict[_InputRef, _Array],
        replace: Dict[_OutputRef, List[_InputArgument]],
    ) -> Tuple[_Operation, List[_Array]]:
        """Add an operation to the model.

        !!! note

            This function assumes that each input array precedes every consumer operation; i.e., there is no need to
            reorder `fx.Node`s/instructions.

        Args:
            target: The label for the operation.
            operation: The FX IR operation to be added to the model.
            inputs: For each input of the new operation, the array that it reads.
            replace: For each output of the new operation, specify which downstream operation inputs to replace.

        Returns:
            A pair whose first item is the new operation node, and whose second item is the collection of array nodes
                produced by the operation node.

        """
        self._check_target(target)
        if self.debug:
            self._validate_add_operation_inputs(target, operation, inputs)
            self._validate_add_operation_replace(target, operation, replace)

        # select the insertion point
        # TODO: This block assumes that each input array precedes every consumer operation.
        #       Implement a method to reorder `fx.Node`s/instructions
        if len(inputs) == 0:
            anchor, = self.graph_module.graph.find_nodes(op="placeholder")
        else:
            # TODO: This branch has time complexity O(|V|).
            #       Cache the `line_no` of existing arrays to make it O(1).
            node_to_line_no = {node: line_no for line_no, node in enumerate(self.graph_module.graph.nodes)}
            anchor = max(list(inputs.values()), key=lambda node: node_to_line_no[node])
        # pre-process the `replace` argument, making it amenable for replacing the inputs of downstream operations
        input_argument_to_output_refs = collections.defaultdict(list)
        for output_ref, input_arguments in replace.items():
            for arg in input_arguments:
                input_argument_to_output_refs[arg].append([output_ref])
        # update the model
        arity, co_arity = arity_and_co_arity.ARITY_AND_CO_ARITY[type(operation)]
        with self.graph_module.graph.inserting_after(anchor):
            self.graph_module.register_module(target, operation)
            node = self.graph_module.graph.call_module(target, ())
            node.args = tuple(inputs[(i, None)] for i in range(arity))
            self.operations[node] = None
            for input_ in node.args:
                self.reads[input_] += 1
        anchor = node
        outputs = []
        for j in range(co_arity):
            with self.graph_module.graph.inserting_after(anchor):
                self.graph_module.register_module(f"{target}_{j}", operators.Array(j))
                array = self.graph_module.graph.call_module(f"{target}_{j}", (node,))
                outputs.append(array)
                self.reads[array] = 0
            anchor = array
        for (input_idx, cons), ((_, output_idx),) in input_argument_to_output_refs.items():
            old_array, new_array = cons.args[input_idx], outputs[output_idx]
            cons.update_arg(input_idx, outputs[output_idx])
            self.reads[old_array] -= 1
            self.reads[new_array] += 1

        return node, outputs

    def _validate_remove_operation_replace(
        self, node: _Operation, replace: Dict[_OutputRef, Dict[_InputArgument, _Array]]
    ) -> None:
        """Check whether the `replace` argument to a `remove_operation` call is valid.

        Args:
            node: The operation node to remove.
            replace: For each output of the operation, specify how to replace each of its uses with another array.

        Raises:
            ValueError: The `replace` argument does not specify how to replace an operation output.
            ValueError: The `replace` argument specifies how to replace an array that is not an operation output.
            ValueError: The `replace` argument specifies the input of a downstream operation incorrectly as an operation output.
            ValueError: The `replace` argument specifies more than one way to replace the input of a downstream operation.
            ValueError: Not all the uses of an operation output are replaced.

        """
        operation = self.graph_module.get_submodule(node.target)
        _, co_arity = arity_and_co_arity.ARITY_AND_CO_ARITY[type(operation)]
        actual, expected = {j for _, j in replace}, {j for j in range(co_arity)}
        underspecified, overspecified = expected - actual, actual - expected
        if underspecified:
            raise ValueError(f"Replacements of outputs {underspecified} of operation {node.target} are not specified")
        if overspecified:
            raise ValueError(f"Replacements of outputs {overspecified} of operation {node.target} are not required")

        for (_, output_idx), input_argument_to_array in replace.items():
            for (input_idx, cons), _ in input_argument_to_array.items():
                if not (cons.args[input_idx] is node):
                    raise ValueError(f"{output_idx}-th output of operation {node.target} is not {input_idx}-th input of operation {cons.target}")

        input_argument_to_arrays = collections.defaultdict(list)
        for (_, output_idx), input_argument_to_array in replace.items():
            for input_argument, array in input_argument_to_array.items():
                input_argument_to_arrays[input_argument].append(array)
        for (input_idx, cons), arrays in input_argument_to_arrays.items():
            if len(arrays) > 1:
                raise ValueError(f"Replacement of {input_idx}-th input of operation {cons.target} is ambiguous")

        outputs = list(node.users)
        for (_, output_idx), input_argument_to_array in replace.items():
            before_n_uses, after_n_uses = len(outputs[output_idx].users), len(input_argument_to_array)
            if after_n_uses != before_n_uses:
                raise ValueError(f"{output_idx}-th output of operation {node.target} is used {before_n_uses} times but replaced {after_n_uses} times")

    def remove_operation(self, node: _Operation, replace: Dict[_OutputRef, Dict[_InputArgument, _Array]]) -> None:
        """Remove an operation from the model.

        Args:
            node: The operation node to remove.
            replace: For each output of the operation, specify how to replace each of its uses with another array.

        Warns:
            UserWarning: An operation output is a model output.

        """
        if self.debug:
            self._validate_remove_operation_replace(node, replace)

        # validate the operation to remove
        outputs = [array for array in node.users if (array in self.outputs)]
        if outputs:
            warnings.warn(f"Outputs {outputs} are model outputs", UserWarning)
            return
        # update the model
        for (_, output_idx), input_argument_to_array in replace.items():
            for (input_idx, cons), array in input_argument_to_array.items():
                old_array, new_array = cons.args[input_idx], array
                cons.replace_arg(input_idx, array)
                self.reads[old_array] -= 1
                self.reads[new_array] += 1

        outputs = list(node.users)
        for array in outputs[::-1]:
            self.graph_module.graph.erase_node(array)
            self.graph_module.delete_submodule(array.target)
            del self.reads[array]
        for array in node.args:
            self.reads[array] -= 1
        self.graph_module.graph.erase_node(node)
        self.graph_module.delete_submodule(node.target)
        del self.operations[node]

    def finalize(self) -> fx.GraphModule:
        """Consolidate the model and return it to the user.

        Returns:
            The model.

        Warns:
            UserWarning: The model has orphaned arrays (debug-mode-only).

        """
        self.graph_module.graph.lint()
        self.graph_module.recompile()
        validator.VALIDATOR.validate(self.graph_module)
        if self.debug and self.orphaned:
            warnings.warn(f"Arrays {self.orphaned} are orphaned")
        return self.graph_module
