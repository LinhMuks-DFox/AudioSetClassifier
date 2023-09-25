import torch
import typing


def label_digit2tensor(label: typing.List[int], class_num=527) -> torch.Tensor:
    t = torch.zeros(class_num)
    idx = torch.tensor(label)
    t[idx] = 1
    return t
