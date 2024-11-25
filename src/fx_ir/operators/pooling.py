from typing import Tuple

import torch
import torch.nn as nn


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        ys = super().forward(x),
        return ys


class MaxPool2d(nn.MaxPool2d):

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        ys = super().forward(x),
        return ys
