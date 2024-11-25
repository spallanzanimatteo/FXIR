import operator
from typing import Callable, Dict

import torch
import torch.fx as fx

import fx_ir.converter.fx.operations as operations
import fx_ir.operators as operators


def _convert_operator_add(src_operation: operations.CallFunctionOperation, graph_module: fx.GraphModule) -> operators.Add:
    assert src_operation.function is operator.add
    tgt_operation = operators.Add()
    return tgt_operation


def _convert_torch_flatten(src_operation: operations.CallFunctionOperation, graph_module: fx.GraphModule) -> operators.Flatten:
    assert src_operation.function is torch.flatten
    args, kwargs = src_operation.node.normalized_arguments(root=graph_module)
    match len(args):
        case 1:
            dim_start, dim_end = kwargs["start_dim"], kwargs["end_dim"]
        case 2:
            dim_start, dim_end = args[1], kwargs["end_dim"]
        case 3:
            dim_start, dim_end = args[1], args[2]
        case _:
            raise ValueError
    tgt_operation = operators.Flatten(dim_start, dim_end)
    return tgt_operation


_FUNCTION_TO_CONVERTER: Dict[Callable, Callable[[operations.CallFunctionOperation, fx.GraphModule], operators.OperatorT]] = {
    operator.add: _convert_operator_add,
    torch.flatten: _convert_torch_flatten,
}


def convert_call_function(src_operation: operations.CallFunctionOperation, graph_module: fx.GraphModule) -> operators.OperatorT:
    converter = _FUNCTION_TO_CONVERTER[src_operation.function]
    tgt_operation = converter(src_operation, graph_module)
    return tgt_operation
