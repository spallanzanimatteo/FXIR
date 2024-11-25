import functools

import torch
import torch.nn as nn

import fx_ir.operators as operators


@functools.singledispatch
def convert_call_module(src_operation: nn.Module) -> operators.OperatorT:
    raise NotImplementedError


@convert_call_module.register
def _(src_operation: nn.AdaptiveAvgPool2d) -> operators.AdaptiveAvgPool2d:
    tgt_operation = operators.AdaptiveAvgPool2d(output_size=src_operation.output_size)
    return tgt_operation


@convert_call_module.register
def _(src_operation: nn.BatchNorm2d) -> operators.BatchNorm2d:
    tgt_operation = operators.BatchNorm2d(
        num_features=src_operation.num_features,
        eps=src_operation.eps,
        momentum=src_operation.momentum,
        affine=src_operation.affine,
        track_running_stats=src_operation.track_running_stats,
    )
    with torch.no_grad():
        tgt_operation.weight.copy_(src_operation.weight)
        if tgt_operation.affine:
            tgt_operation.bias.copy_(src_operation.bias)
    return tgt_operation


@convert_call_module.register
def _(src_operation: nn.Conv2d) -> operators.Conv2d:
    tgt_operation = operators.Conv2d(
        in_channels=src_operation.in_channels,
        out_channels=src_operation.out_channels,
        kernel_size=src_operation.kernel_size,
        stride=src_operation.stride,
        padding=src_operation.padding,
        dilation=src_operation.dilation,
        groups=src_operation.groups,
        bias=src_operation.bias,
        padding_mode=src_operation.padding_mode,
    )
    with torch.no_grad():
        tgt_operation.weight.copy_(src_operation.weight)
        if not (tgt_operation.bias is None):
            tgt_operation.bias.copy_(src_operation.bias)
    return tgt_operation


@convert_call_module.register
def _(src_operation: nn.Linear) -> operators.Linear:
    tgt_operation = operators.Linear(
        in_features=src_operation.in_features,
        out_features=src_operation.out_features,
        bias=not (src_operation.bias is None),
    )
    with torch.no_grad():
        tgt_operation.weight.copy_(src_operation.weight)
        if not (tgt_operation.bias is None):
            tgt_operation.bias.copy_(src_operation.bias)
    return tgt_operation


@convert_call_module.register
def _(src_operation: nn.MaxPool2d) -> operators.MaxPool2d:
    tgt_operation = operators.MaxPool2d(
        kernel_size=src_operation.kernel_size,
        stride=src_operation.stride,
        padding=src_operation.padding,
        dilation=src_operation.dilation,
    )
    return tgt_operation


@convert_call_module.register
def _(src_operation: nn.ReLU) -> operators.ReLU:
    tgt_operation = operators.ReLU()
    return tgt_operation
