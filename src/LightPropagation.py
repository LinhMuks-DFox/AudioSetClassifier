from typing import Union
import torch
import torch.nn as nn


class LightPropagation(nn.Module):
    def __init__(self, distance: float, bias: Union[float, None], std: float):
        super(LightPropagation, self).__init__()
        self.distance = distance
        self.bias = bias
        self.std = std

    def forward(self, x):
        attenuation = 1 / (self.distance ** 2)
        if self.bias is None:
            bias = torch.rand(1).to(x.device)
        else:
            bias = self.bias
        noise = self.std * torch.randn(x.shape).to(x.device)
        x = attenuation * x + bias + noise
        return x
