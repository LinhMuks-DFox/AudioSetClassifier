import typing

import torch


def label_digit2tensor(label: typing.List[int], class_num=527) -> torch.Tensor:
    t = torch.zeros(class_num)
    idx = torch.tensor(label)
    t[idx] = 1
    return t


def fix_length(audio_data: torch.Tensor, sample_length: int) -> torch.Tensor:
    return torch.nn.functional.pad(audio_data, (0, sample_length - audio_data.shape[1]))
