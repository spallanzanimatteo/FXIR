import networkx as nx
import torch.fx as fx

import fx_ir.operators as opset


_AR_BLOCK_ID = 0
"""The identifier of the graph's array block."""

_OP_BLOCK_ID = 1
"""The identifier of the graph's operation block."""


def _from_fx_to_nx(model: fx.GraphModule) -> nx.MultiDiGraph:
    """Convert a `GraphModule` to a NetworkX representation.

    Args:
        model: A `GraphModule`.

    Returns:
        The NetworkX representation of the `GraphModule`.

    """
    nx_graph = nx.MultiDiGraph()
    for fx_node in model.graph.nodes:
        if (fx_node.op == "call_module") and isinstance(model.get_submodule(fx_node.target), opset.Array):
            bipartite = _AR_BLOCK_ID
        else:
            bipartite = _OP_BLOCK_ID
        nx_graph.add_node(fx_node.target, opcode=fx_node.op, bipartite=bipartite)
        for fx_arg in fx_node.args:
            if fx_node.op != "output":
                fx_arg = (fx_arg,)  # canonicalize the argument to a container
            for fx_arg in fx_arg:
                nx_graph.add_edge(fx_arg.target, fx_node.target)

    return nx_graph


def _are_fx_opcodes_valid(nx_graph: nx.MultiDiGraph) -> None:
    """Validate whether the FX opcodes of a `GraphModule` are valid FX IR opcodes.

    Args:
        nx_graph: The NetworkX representation of a `GraphModule`.

    """
    opcodes = [nx_graph.nodes[n]["opcode"] for n in nx_graph.nodes]
    assert opcodes.count("placeholder") == 1
    assert opcodes.count("output") == 1
    assert opcodes.count("call_module") == (len(opcodes) - 2)


def _is_bipartite(nx_graph: nx.MultiDiGraph) -> None:
    """Validate whether a `GraphModule` is bipartite.

    Args:
        nx_graph: The NetworkX representation of a `GraphModule`.

    """
    assert nx.is_bipartite(nx_graph)


def validate(model: fx.GraphModule) -> None:
    """Validate whether a `GraphModule` satisfies the invariants of the FX IR.

    Args:
        model: A `GraphModule`.

    """
    nx_graph = _from_fx_to_nx(model)
    _are_fx_opcodes_valid(nx_graph)
    _is_bipartite(nx_graph)
