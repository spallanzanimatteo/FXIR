import inspect
from typing import Type, get_args

import torch

import fx_ir.operators as operators


def _get_operator_arity(operator: Type[operators.OperatorT]) -> int:
    """Return the arity of an FX IR operator.

    Args:
        operator: An FX IR operator.

    Returns:
        The number of arrays consumed by the operator.

    """
    return len([p.annotation for p in inspect.signature(operator.forward).parameters.values() if (p.annotation is torch.Tensor)])


def _get_operator_co_arity(operator: Type[operators.OperatorT]) -> int:
    """Return the co-arity of an FX IR operator.

    Args:
        operator: An FX IR operator.

    Returns:
        The number of arrays produced by the operator.

    """
    return len([r for r in get_args(inspect.signature(operator.forward).return_annotation) if (r is torch.Tensor)])


ARITY_AND_CO_ARITY = {
    operator: (_get_operator_arity(operator), _get_operator_co_arity(operator)) for operator in operators.Operators
}
"""Map each FX IR operator to its arity and co-arity."""
