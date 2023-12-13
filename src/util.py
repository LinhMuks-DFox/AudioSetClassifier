import typing

import numpy as np
import torch


def label_digit2tensor(label_digits: typing.List[int], class_num=527) -> torch.Tensor:
    label_digits: np.ndarray = np.array(label_digits)
    label: np.ndarray = np.zeros(class_num)
    label[label_digits] = 1
    return torch.tensor(label)


def fix_length(audio_data: torch.Tensor, sample_length: int) -> torch.Tensor:
    return torch.nn.functional.pad(audio_data, (0, sample_length - audio_data.shape[1]))


def blinky_data_normalize(data: torch.Tensor):
    data_min = torch.min(data)
    de_min = data - data_min
    return de_min / torch.max(de_min)
