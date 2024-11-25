import collections
from typing import Dict, Type

import torch.fx as fx

import fx_ir.converter.fx.operations as operations
import fx_ir.converter.fx.converter.convert_call_function as convert_call_function
import fx_ir.converter.fx.converter.convert_call_module as convert_call_module
import fx_ir.editor as editor
import fx_ir.operators as operators

_SrcInput = fx.Node

_SrcOperation = fx.Node

_Array = fx.Node

_Operation = fx.Node

_TgtInput = _Array

_TgtOperation = _Operation


class FXConverter:

    def __init__(self, debug: bool) -> None:
        self.editor = editor.Editor(debug=debug)

    def toggle(self) -> None:
        self.editor.toggle()

    def convert(self, graph_module: fx.GraphModule) -> fx.GraphModule:
        tgt_op_type_to_count: collections.Counter[Type[operators.OperatorT]] = collections.Counter()

        src_in_node_to_tgt_in_node: Dict[_SrcInput, _TgtInput] = {}
        src_op_node_to_tgt_op_node: Dict[_SrcOperation, _TgtOperation] = {}

        for node in graph_module.graph.nodes:
            match node.op:
                case "placeholder":
                    src_op = operations.InputOperation(opcode=node.op, node=node)
                    src_in_node, tgt_in_node = src_op.node, self.editor.add_input()
                    src_in_node_to_tgt_in_node[src_in_node] = tgt_in_node
                case "output":
                    src_op = operations.OutputOperation(opcode=node.op, node=node)
                    src_op_node = src_op.node
                    src_out_nodes = src_op_node.args[0] if isinstance(src_op_node.args[0], tuple) else (src_op_node.args[0],)
                    for src_out_node in src_out_nodes:
                        tgt_out_node = next(iter(src_op_node_to_tgt_op_node[src_out_node].users))
                        self.editor.add_output(tgt_out_node)
                case "call_module":
                    src_op = operations.CallModuleOperation(opcode=node.op, node=node, module=graph_module.get_submodule(node.target))
                    src_op_node = src_op.node
                    tgt_op = convert_call_module.convert_call_module(src_op.module)
                    tgt_op_type_to_count[type(tgt_op)] += 1
                    new_args = {}
                    for i, arg in enumerate(src_op_node.args):
                        if arg in src_op_node_to_tgt_op_node:
                            new_args[(i, None)] = next(iter(src_op_node_to_tgt_op_node[arg].users))
                        if arg in src_in_node_to_tgt_in_node:
                            new_args[(i, None)] = src_in_node_to_tgt_in_node[arg]
                    tgt_op_node, arr_nodes = self.editor.add_operation(
                        self.editor.output,
                        f"{type(tgt_op).__name__}_{tgt_op_type_to_count[type(tgt_op)]}",
                        tgt_op,
                        new_args,
                        {},
                    )
                    src_op_node_to_tgt_op_node[src_op_node] = tgt_op_node
                case "call_method":
                    raise NotImplementedError
                case "call_function":
                    src_op = operations.CallFunctionOperation(opcode=node.op, node=node)
                    src_op_node = src_op.node
                    tgt_op = convert_call_function.convert_call_function(src_op, graph_module)
                    tgt_op_type_to_count[type(tgt_op)] += 1
                    new_args = {}
                    for i, arg in enumerate(src_op_node.args):
                        if arg in src_op_node_to_tgt_op_node:
                            new_args[(i, None)] = next(iter(src_op_node_to_tgt_op_node[arg].users))
                        if arg in src_in_node_to_tgt_in_node:
                            new_args[(i, None)] = src_in_node_to_tgt_in_node[arg]
                    tgt_op_node, arr_nodes = self.editor.add_operation(
                        self.editor.output,
                        f"{type(tgt_op).__name__}_{tgt_op_type_to_count[type(tgt_op)]}",
                        tgt_op,
                        new_args,
                        {},
                    )
                    src_op_node_to_tgt_op_node[src_op_node] = tgt_op_node
                case "get_attr":
                    raise NotImplementedError
                case _:
                    raise ValueError

        graph_module = self.editor.finalize()
        self.editor.reset()
        return graph_module
