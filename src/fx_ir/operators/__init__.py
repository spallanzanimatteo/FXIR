from typing import Union, get_args

from fx_ir.operators.activation import ReLU
from fx_ir.operators.arithmetic import Add, Mul
from fx_ir.operators.array import Array
from fx_ir.operators.linear import Linear
from fx_ir.operators.memory import Concat, Split

OperatorT = Union[
    Add,
    Concat,
    Linear,
    Mul,
    ReLU,
    Split,
]
Operators = get_args(OperatorT)

del Union, get_args
