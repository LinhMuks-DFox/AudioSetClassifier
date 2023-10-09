import typing

import torch
import numpy as np


def label_digit2tensor(label_digits: typing.List[int], class_num=527) -> torch.Tensor:
    label_digits: np.ndarray = np.array(label_digits)
    label: np.ndarray = np.zeros(class_num, dtype=np.int32)
    label[label_digits] = 1
    return torch.tensor(label)


def fix_length(audio_data: torch.Tensor, sample_length: int) -> torch.Tensor:
    return torch.nn.functional.pad(audio_data, (0, sample_length - audio_data.shape[1]))
