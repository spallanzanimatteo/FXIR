from typing import Any, Dict, List, Tuple, Union

import torch
import torch.fx as fx


def _compute_type_expression(obj: Any) -> Any:
    if isinstance(obj, dict):
        k, v = next(iter(obj.items()))
        type_expression = Dict[_compute_type_expression(k), _compute_type_expression(v)]
    elif isinstance(obj, list):
        type_expression = List[*(_compute_type_expression(item) for item in obj)]
    elif isinstance(obj, tuple):
        type_expression = Tuple[*(_compute_type_expression(item) for item in obj)]
    elif isinstance(obj, torch.Tensor):
        type_expression = torch.Tensor
    else:
        type_expression = type(obj)
    return type_expression


def _unpack_container_arg(env: Dict[Tuple[str, str], Any], arg: Any) -> Any:
    if isinstance(arg, dict):
        arg = {_unpack_container_arg(env, k): _unpack_container_arg(env, v) for k, v in arg.items()}
    elif isinstance(arg, (list, tuple)):
        arg = type(arg)((_unpack_container_arg(env, item) for item in arg))
    elif isinstance(arg, fx.Node):
        arg = env[(arg.op, arg.name)]
    else:
        return arg
    return arg


def infer_type(graph_module: fx.GraphModule, input_: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]) -> None:
    env: Dict[Tuple[str, str], Any] = {}
    for node in graph_module.graph.nodes:
        match node.op:
            case "placeholder":
                output = input_[node.target]
                node.type = _compute_type_expression(output)
                env[(node.op, node.name)] = output
            case "output":
                node.type = None
            case "call_module":
                args = tuple(_unpack_container_arg(env, arg) for arg in node.args)
                kwargs = {k: _unpack_container_arg(env, v) for k, v in node.kwargs.items()}
                output = graph_module.get_submodule(node.target)(*args, **kwargs)
                node.type = _compute_type_expression(output)
                env[(node.op, node.name)] = output
            case "call_method":
                raise NotImplementedError
            case "call_function":
                args = tuple(_unpack_container_arg(env, arg) for arg in node.args)
                kwargs = {k: _unpack_container_arg(env, v) for k, v in node.kwargs.items()}
                output = node.target(*args, **kwargs)
                node.type = _compute_type_expression(output)
                env[(node.op, node.name)] = output
            case "get_attr":
                raise NotImplementedError
            case _:
                raise ValueError
