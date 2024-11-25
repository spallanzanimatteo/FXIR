from typing import Union, get_args

from fx_ir.operators.activation import ReLU
from fx_ir.operators.arithmetic import Add, Mul
from fx_ir.operators.array import Array
from fx_ir.operators.convolution import Conv2d
from fx_ir.operators.linear import Linear
from fx_ir.operators.memory import Concat, Flatten, Split
from fx_ir.operators.normalization import BatchNorm2d
from fx_ir.operators.pooling import AdaptiveAvgPool2d, MaxPool2d

OperatorT = Union[
    AdaptiveAvgPool2d,
    Add,
    BatchNorm2d,
    Concat,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Mul,
    ReLU,
    Split,
]
Operators = get_args(OperatorT)

del Union, get_args
