import torch
import typing


def label_digit2tensor(label: typing.List[int]) -> torch.Tensor:
    t = torch.zeros(527)
    idx = torch.tensor(label)
    t[idx] = 1
    return t
