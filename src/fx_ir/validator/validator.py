import enum

import networkx as nx
import torch.fx as fx

import fx_ir.operators as operators


class _Partition(enum.Enum):
    """Label the array and operation nodes of a graph."""
    ARRAY = enum.auto()
    OPERATION = enum.auto()


class _Validator:
    """Validate FX IR compliance."""

    def _check_opcodes(self, model: fx.GraphModule) -> None:
        """Check whether the FX opcodes of a model are valid FX IR opcodes.

        Args:
            model: A model.

        Raises:
            ValueError: The model has no input or more than one input.
            ValueError: The model has no output or more than one output.
            ValueError: The model contains non-"call_module" nodes other than the input and output nodes.

        """
        opcodes = [node.op for node in model.graph.nodes]

        n_inputs, n_outputs, n_call_modules = opcodes.count("placeholder"), opcodes.count("output"), opcodes.count("call_module")
        if n_inputs != 1:
            raise ValueError(f"Model has {n_inputs} inputs (expected {1})")
        if n_outputs != 1:
            raise ValueError(f"Model has {n_outputs} outputs (expected {1})")
        if n_call_modules != (len(opcodes) - 2):
            raise ValueError(f"Model has {n_call_modules} outputs (expected {len(opcodes) - 2})")

    def _check_call_modules(self, model: fx.GraphModule) -> None:
        """Check whether all "call_module" nodes are either FX IR arrays or FX IR operators.

        Args:
            model: A model.

        Raises:
            TypeError: A module is not an FX IR array or operator.

        """
        nodes = model.graph.find_nodes(op="call_module")
        for node in nodes:
            module = model.get_submodule(node.target)
            if not isinstance(module, operators.Operators + (operators.Array,)):
                raise TypeError(f"Module {node.target} is not an FX IR array or operator: {type(module).__name__}")

    def _check_composition(self, model: fx.GraphModule) -> None:
        """Check whether a model satisfies the opcode and composition invariants of the FX IR.

        Args:
            model: A model.

        """
        self._check_opcodes(model)
        self._check_call_modules(model)

    def _from_fx_to_nx(self, model: fx.GraphModule) -> nx.MultiDiGraph:
        """Convert an `fx.GraphModule` to an `nx.MultiDiGraph`.

        Args:
            model: An `fx.GraphModule`.

        Returns:
            The `nx.MultiDiGraph` representation of the `fx.GraphModule`.

        """
        nx_graph = nx.MultiDiGraph()
        for fx_node in model.graph.nodes:
            if fx_node.op != "call_module":
                continue
            module = model.get_submodule(fx_node.target)
            bipartite = _Partition.ARRAY if isinstance(module, operators.Array) else _Partition.OPERATION
            nx_graph.add_node(fx_node.target, opcode=fx_node.op, bipartite=bipartite)
            for fx_arg in fx_node.args:
                nx_graph.add_edge(fx_arg.target, fx_node.target)

        return nx_graph

    def _check_bipartiteness(self, model: fx.GraphModule) -> None:
        """Check whether the graph of a model is bipartite.

        Args:
            model: A model.

        Raises:
            ValueError: The model's graph is not bipartite.

        """
        nx_graph = self._from_fx_to_nx(model)
        if not nx.is_bipartite(nx_graph):
            raise ValueError("Model graph is not bipartite")

    def validate(self, model: fx.GraphModule) -> None:
        """Check whether a model satisfies the invariants of the FX IR.

        Args:
            model: A model.

        """
        self._check_composition(model)
        self._check_bipartiteness(model)


VALIDATOR = _Validator()
"""The unique model validator instance."""
