import dataclasses
import functools
from typing import Callable, List, Literal, Union

import torch.fx as fx
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class _SourceOperation:
    opcode: Literal["placeholder", "output", "call_module", "call_function"]
    node: fx.Node


@dataclasses.dataclass(frozen=True)
class InputOperation(_SourceOperation):
    opcode: Literal["placeholder"]


@dataclasses.dataclass(frozen=True)
class OutputOperation(_SourceOperation):
    opcode: Literal["output"]


@dataclasses.dataclass(frozen=True)
class CallModuleOperation(_SourceOperation):
    opcode: Literal["call_module"]
    module: nn.Module

    @functools.cached_property
    def target(self) -> str:
        return self.node.target


@dataclasses.dataclass(frozen=True)
class CallFunctionOperation(_SourceOperation):
    opcode: Literal["call_function"]

    @functools.cached_property
    def target(self) -> str:
        return self.node.name

    @functools.cached_property
    def function(self) -> Callable:
        return self.node.target


SourceOperation = Union[InputOperation, OutputOperation, CallModuleOperation, CallFunctionOperation]


def find_src_operations(graph_module: fx.GraphModule) -> List[SourceOperation]:
    src_operations = []
    operation: SourceOperation
    for node in graph_module.graph.nodes:
        match node.op:
            case "placeholder":
                operation = InputOperation(opcode=node.op, node=node)
            case "output":
                operation = OutputOperation(opcode=node.op, node=node)
            case "call_module":
                operation = CallModuleOperation(opcode=node.op, node=node, module=graph_module.get_submodule(node.target))
            case "call_method":
                raise NotImplementedError
            case "call_function":
                operation = CallFunctionOperation(opcode=node.op, node=node)
            case "get_attr":
                raise NotImplementedError
            case _:
                raise ValueError
        src_operations.append(operation)

    return src_operations
