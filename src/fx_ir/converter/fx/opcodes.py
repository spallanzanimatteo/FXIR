import dataclasses
from typing import Callable, List, Literal, Type, Union
from typing_extensions import Self

import torch.nn as nn
import torch.fx as fx

import fx_ir.converter.fx.operations as operations


@dataclasses.dataclass(frozen=True, eq=True)
class _SourceOpcode:
    opcode: Literal["placeholder", "output", "call_module", "call_function"]


@dataclasses.dataclass(frozen=True, eq=True)
class InputOpcode(_SourceOpcode):
    opcode: Literal["placeholder"]

    @classmethod
    def from_operation(cls, op: operations.InputOperation) -> Self:
        return cls(opcode=op.opcode)


@dataclasses.dataclass(frozen=True, eq=True)
class OutputOpcode(_SourceOpcode):
    opcode: Literal["output"]

    @classmethod
    def from_operation(cls, op: operations.OutputOperation) -> Self:
        return cls(opcode=op.opcode)


@dataclasses.dataclass(frozen=True, eq=True)
class CallModuleOpcode(_SourceOpcode):
    opcode: Literal["call_module"]
    type: Type[nn.Module]

    @classmethod
    def from_operation(cls, op: operations.CallModuleOperation) -> Self:
        return cls(opcode=op.opcode, type=type(op.module))


@dataclasses.dataclass(frozen=True, eq=True)
class CallFunctionOpcode(_SourceOpcode):
    opcode: Literal["call_function"]
    function: Callable

    @classmethod
    def from_operation(cls, op: operations.CallFunctionOperation) -> Self:
        return cls(opcode=op.opcode, function=op.function)


SourceOpcode = Union[InputOpcode, OutputOpcode, CallModuleOpcode, CallFunctionOpcode]


def find_src_opcodes(graph_module: fx.GraphModule) -> List[SourceOpcode]:
    src_opcodes = set()
    opcode: SourceOpcode
    for node in graph_module.graph.nodes:
        match node.op:
            case "placeholder":
                opcode = InputOpcode(opcode=node.op)
            case "output":
                opcode = OutputOpcode(opcode=node.op)
            case "call_function":
                opcode = CallFunctionOpcode(opcode=node.op, function=node.target)
            case "call_method":
                raise NotImplementedError
            case "call_module":
                opcode = CallModuleOpcode(opcode=node.op, type=type(graph_module.get_submodule(node.target)))
            case "get_attr":
                raise NotImplementedError
            case _:
                raise ValueError
        src_opcodes.add(opcode)

    return list(src_opcodes)
