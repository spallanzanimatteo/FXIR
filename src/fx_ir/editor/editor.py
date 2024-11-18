import collections
from typing import Dict, List, Mapping, Tuple, Union
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

_Input = fx.Node
"""A 'placeholder' node representing the model input."""

_Output = fx.Node
"""An 'output' node representing the model output."""

_Array = fx.Node
"""A 'call_module' node representing an array."""

_Operation = fx.Node
"""A 'call_module' node representing an operation."""

_InputReference = Tuple[int, _Operation]
"""A reference to an input argument to an operation node. Can be resolved to a 'call_module' node representing an array."""

_OutputReference = Tuple[_Operation, int]
"""A reference to an output value of an operation node. Can be resolved to a 'call_module' node representing an array."""

_InputReferenceStub = Tuple[int, None]
"""A reference to an input argument to an operation node that has not yet been created. Can be resolved to an array node once the operation node has been created."""

_OutputReferenceStub = Tuple[None, int]
"""A reference to an output value of an operation node that has not yet been created. Can be resole to an array node once the operation node has been created."""

_AddInputsMap = Mapping[_InputReferenceStub, _Array]
"""Map a reference to an input argument of an operation that has not been created to the array that it reads."""

_AddReplacementMap = Mapping[_InputReference, _OutputReferenceStub]
"""Map a reference to an input argument of a downstream operation to a reference to the output values of an operation node that has not yet been created. To be used when adding a new operation."""

_RemoveReplacementMap = Mapping[_InputReference, _Array]
"""Map a reference to an input argument of a downstream operation to the array that it reads. To be used when removing an existing operation."""


class Editor:
    """Create and edit an FX-IR-compliant model."""

    def __init__(self, debug: bool) -> None:
        """Initialize the editor.

        Args:
            debug: Whether to perform exhaustive sanity checks and output graph status messages during editing.

        Attributes:
            graph_module: The model being edited.
            inputs: The arrays that are inputs to the model.
            outputs: The arrays that are outputs of the model.
            reads: For each array, how many times it is read.
            operations: The operations of the model.

        """
        self.debug = debug
        self.graph_module: fx.GraphModule = _create_empty_graph_module()
        self.inputs: Dict[_Array, None] = dict()
        self.outputs: Dict[_Array, None] = dict()
        self.reads: collections.Counter[_Array] = collections.Counter()
        self.operations: Dict[_Operation, None] = dict()

    @property
    def input(self) -> _Input:
        """Return the node representing the model's input.

        Returns:
            The node representing the model's input.

        """
        in_, = self.graph_module.graph.find_nodes(op="placeholder")
        return in_

    @property
    def output(self) -> _Output:
        """Return the node representing the model's output.

        Returns:
            The node representing the model's output.

        """
        out, = self.graph_module.graph.find_nodes(op="output")
        return out

    @property
    def orphaned(self) -> List[_Array]:
        """Return the arrays that are neither read by any operation, nor model outputs.

        Returns:
            The arrays that are neither read by any operation, nor model outputs. They are dead code.

        """
        orphaned = [arr_node for arr_node, count in self.reads.items() if ((count == 0) and (not (arr_node in self.outputs)))]
        return orphaned

    def toggle(self) -> None:
        """Switch from debug to production mode, and vice versa."""
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

    def add_output(self, arr_node: _Array) -> None:
        """Add an output to the model.

        Args:
            arr_node: The new output of the model.

        Warns:
            UserWarning: The node is not an array.

        """
        if arr_node in self.outputs:
            return

        # validate the candidate output
        if not isinstance(self.graph_module.get_submodule(arr_node.target), operators.Array):
            warnings.warn(f"Node {arr_node.target} is not an array", UserWarning)
            return

        # prepare the model update
        out = self.output
        out_nodes = out.args[0]
        out_nodes += (arr_node,)
        # update the model
        out.update_arg(0, out_nodes)
        self.outputs[arr_node] = None

    def remove_output(self) -> None:
        """Remove the last output from the model.

        Warns:
            UserWarning: The array is orphaned (debug-mode-only).

        """
        if len(self.outputs) == 0:
            return

        # prepare the model update
        out = self.output
        out_nodes = out.args[0]
        out_nodes, (arr_node,) = out_nodes[0:-1], out_nodes[-1:]
        # update the model
        del self.outputs[arr_node]
        out.update_arg(0, out_nodes)

        if self.debug and (self.reads[arr_node] == 0):
            warnings.warn(f"Array {arr_node.target} is orphaned", UserWarning)

    def add_input(self) -> _Array:
        """Add an input to the model.

        Returns:
            The new input to the model.

        Warns:
            UserWarning: The array is orphaned (debug-mode-only).

        """
        # prepare the model update
        in_ = self.input
        in_nodes, idx = tuple(in_.users), len(in_.users)
        target, arr_module = f"{in_.name}_{idx}", operators.Array(idx)
        anchor_node = in_nodes[-1] if in_nodes else in_
        self._check_target(target)
        # update the model
        with self.graph_module.graph.inserting_after(anchor_node):
            self.graph_module.add_submodule(target, arr_module)
            arr_node = self.graph_module.graph.call_module(target, (in_,))
            self.inputs[arr_node] = None
            self.reads[arr_node] = 0

        if self.debug and (self.reads[arr_node] == 0):
            warnings.warn(f"Array {arr_node.target} is orphaned", UserWarning)

        return arr_node

    def remove_input(self) -> None:
        """Remove the last input from the model.

        Warns:
            UserWarning: The input has uses.

        """
        # prepare the model update
        in_ = self.input
        in_nodes = tuple(in_.users)
        in_nodes, (arr_node,) = in_nodes[0:-1], in_nodes[-1:]

        # validate the model update
        if self.reads[arr_node] > 0:
            cons_nodes = list(arr_node.users)
            warnings.warn(f"Array {arr_node.target} has uses: {cons_nodes}", UserWarning)
            return

        # update the model
        del self.reads[arr_node]
        del self.inputs[arr_node]
        self.graph_module.graph.erase_node(arr_node)
        self.graph_module.delete_submodule(arr_node.target)

    def _validate_add_inputs_map(
        self, op_target: str, op_module: operators.OperatorT, add_inputs_map: _AddInputsMap
    ) -> None:
        """Check whether the `add_inputs_map` argument to an `add_operation` call is valid.

        Args:
            op_target: The label for the new operation.
            op_module: The new operation.
            add_inputs_map: For each reference to an input argument to the new operation, the upstream array that it reads.

        Raises:
            ValueError: The `add_inputs_map` argument does not specify all the input arguments to the operation.
            ValueError: The `add_inputs_map` argument specifies input arguments that are not required by the operation.

        """
        arity, _ = arity_and_co_arity.ARITY_AND_CO_ARITY[type(op_module)]
        actual, expected = {in_idx for in_idx, _ in add_inputs_map}, {in_idx for in_idx in range(arity)}
        underspecified, overspecified = expected - actual, actual - expected
        if underspecified:
            raise ValueError(f"Inputs {sorted(underspecified)} to operator {op_target} are not specified")
        if overspecified:
            raise ValueError(f"Inputs {sorted(overspecified)} to operator {op_target} are not required")

    def _validate_add_replacement_map(
        self, op_target: str, op_module: operators.OperatorT, add_replacement_map: _AddReplacementMap
    ) -> None:
        """Check whether the `add_replacement_map` argument to an `add_operation` call is valid.

        Args:
            op_target: The label for the new operation.
            op_module: The new operation.
            add_replacement_map: How to replace the input arguments to downstream operations with the output values of the new operation.

        Raises:
            ValueError: The `add_replacement_map` argument specifies uses of an array that is not an output of the operation.

        """
        _, co_arity = arity_and_co_arity.ARITY_AND_CO_ARITY[type(op_module)]
        actual, expected = {out_idx for (_, out_idx) in add_replacement_map.values()}, {out_idx for out_idx in range(co_arity)}
        overspecified = actual - expected
        if overspecified:
            raise ValueError(f"Outputs {sorted(overspecified)} of operation {op_target} are not required")

    def add_operation(
        self,
        anchor_node: Union[_Output, _Operation],
        op_target: str,
        op_module: operators.OperatorT,
        add_inputs_map: _AddInputsMap,
        add_replacement_map: _AddReplacementMap,
    ) -> Tuple[_Operation, List[_Array]]:
        """Add an operation to the model.

        !!! note

            This function assumes that each input array precedes every consumer operation; i.e., there is no need to
            reorder `fx.Node`s/instructions.

        Args:
            anchor_node: The output or operation node before which the new operation node should be inserted.
            op_target: The label for the new operation.
            op_module: The new operation.
            add_inputs_map: For each reference to an input argument to the new operation, the upstream array that it reads.
            add_replacement_map: How to replace the input arguments to downstream operations with the output values of the new operation.

        Returns:
            A pair whose first item is the new operation node, and whose second item is the collection of array nodes
                produced by the operation node.

        Warns:
            UserWarning: An array is orphaned.

        """
        if self.debug:
            self._validate_add_inputs_map(op_target, op_module, add_inputs_map)
            self._validate_add_replacement_map(op_target, op_module, add_replacement_map)

        # update the model
        arity, co_arity = arity_and_co_arity.ARITY_AND_CO_ARITY[type(op_module)]
        # add the new operation
        self._check_target(op_target)
        with self.graph_module.graph.inserting_before(anchor_node):
            in_nodes = tuple(add_inputs_map[(in_idx, None)] for in_idx in range(arity))
            self.graph_module.register_module(op_target, op_module)
            op_node = self.graph_module.graph.call_module(op_target, in_nodes)
            self.operations[op_node] = None
            for arr_node in in_nodes:
                self.reads[arr_node] += 1
        # add the new operation's outputs
        out_nodes = []
        for out_idx in range(co_arity):
            arr_target, arr_module = f"{op_target}_{out_idx}", operators.Array(out_idx)
            with self.graph_module.graph.inserting_before(anchor_node):
                self.graph_module.register_module(arr_target, arr_module)
                arr_node = self.graph_module.graph.call_module(arr_target, (op_node,))
                self.reads[arr_node] = 0
                out_nodes.append(arr_node)
        # update the inputs to consumer operations
        arr_nodes_old, arr_nodes_new = [], out_nodes
        for (in_idx, cons_node), (_, out_idx) in add_replacement_map.items():
            arr_node_old, arr_node_new = cons_node.args[in_idx], arr_nodes_new[out_idx]
            cons_node.update_arg(in_idx, arr_node_new)
            self.reads[arr_node_old] -= 1
            self.reads[arr_node_new] += 1
            if not (arr_node_old in arr_nodes_old):
                arr_nodes_old.append(arr_node_old)

        if self.debug:
            orphaned = [arr_node.target for arr_node in (arr_nodes_old + arr_nodes_new) if ((self.reads[arr_node] == 0) and not (arr_node in self.outputs))]
            warnings.warn(f"Arrays {orphaned} are orphaned", UserWarning)

        return op_node, out_nodes

    def _validate_remove_replacement_map(
        self, op_node: _Operation, remove_replacement_map: _RemoveReplacementMap
    ) -> None:
        """Check whether the `remove_replacement_map` argument to a `remove_operation` call is valid.

        Args:
            op_node: The operation node to remove.
            remove_replacement_map: How to replace the input arguments to consumer operations with upstream arrays.

        Raises:
            ValueError: The `remove_replacement_map` argument does not specify how to replace an input to a consumer operation.
            ValueError: The `remove_replacement_map` argument specifies how to replace an input to a consumer operation that is not an output of the operation to be removed.

        """
        cons_nodes = {cons_node for out_node in op_node.users for cons_node in out_node.users}
        input_refs = {}
        for cons_node in cons_nodes:
            for in_idx, in_node in enumerate(cons_node.args):
                if in_node in op_node.users:
                    input_ref = (in_idx, cons_node)
                    input_refs[input_ref] = None
        actual, expected = {ref for ref in remove_replacement_map}, {ref for ref in input_refs}
        underspecified, overspecified = expected - actual, actual - expected
        if underspecified:
            raise ValueError(f"Replacements of input references {underspecified} are not specified")
        if overspecified:
            raise ValueError(f"Replacements of input references {overspecified} are not required")

    def remove_operation(self, op_node: _Operation, remove_replacement_map: _RemoveReplacementMap) -> None:
        """Remove an operation from the model.

        Args:
            op_node: The operation node to remove.
            remove_replacement_map: How to replace the input arguments to consumer operations with upstream arrays.

        Warns:
            UserWarning: An operation output is a model output.
            UserWarning: An array is orphaned.

        """
        if self.debug:
            self._validate_remove_replacement_map(op_node, remove_replacement_map)

        # validate the operation to remove
        out_nodes = [arr_node.target for arr_node in op_node.users if (arr_node in self.outputs)]
        if out_nodes:
            warnings.warn(f"Outputs {out_nodes} are model outputs", UserWarning)
            return

        # update the model
        # replace the arguments of consumer operations
        for (in_idx, cons_node), arr_node in remove_replacement_map.items():
            arr_node_old, arr_node_new = cons_node.args[in_idx], arr_node
            cons_node.replace_arg(in_idx, arr_node_new)
            self.reads[arr_node_old] -= 1
            self.reads[arr_node_new] += 1
        # remove the operation and its outputs
        in_nodes, out_nodes = op_node.args, tuple(op_node.users)
        # remove the operation's outputs
        for arr_node in out_nodes[::-1]:
            del self.reads[arr_node]
            self.graph_module.graph.erase_node(arr_node)
            self.graph_module.delete_submodule(arr_node.target)
        # remove the operation
        for arr_node in in_nodes:
            self.reads[arr_node] -= 1
        del self.operations[op_node]
        self.graph_module.graph.erase_node(op_node)
        self.graph_module.delete_submodule(op_node.target)

        if self.debug:
            orphaned = [arr_node.target for arr_node in in_nodes if ((self.reads[arr_node] == 0) and (not (arr_node in self.outputs)))]
            warnings.warn(f"Arrays {orphaned} are orphaned", UserWarning)

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
