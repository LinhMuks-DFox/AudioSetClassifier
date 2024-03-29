from typing import Union

import torch
import torch.nn as nn
import torchaudio


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


class CameraResponse(nn.Module):
    def __init__(self, signal_source_sample_rate: int,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(CameraResponse, self).__init__()
        self.frame_rate = frame_rate
        self.temperature = temperature
        self.resample = torchaudio.transforms.Resample(
            orig_freq=signal_source_sample_rate,
            new_freq=frame_rate,
            resampling_method='sinc_interp_hann'
        )

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x = self.resample(x)
        # x = deepy.nn.functional.softstaircase(x, self.levels, self.temperature)
        return x


class LightToCamera(nn.Module):

    def __init__(self,
                 distance: float,
                 bias: Union[float, None],
                 std: float,
                 signal_source_sample_rate: int,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super().__init__()
        self.light_propa = LightPropagation(
            distance=distance, bias=bias, std=std
        )
        self.camera = CameraResponse(
            signal_source_sample_rate=signal_source_sample_rate,
            frame_rate=frame_rate,
            temperature=temperature
        )

    def forward(self, x):
        return self.camera(self.light_propa(x))


if __name__ == "__main__":
    dummy_data = torch.arange(600)
    dummy_data = dummy_data.reshape((4, -1))
    light = LightPropagation(
        distance=5, bias=0.1, std=0.05
    )
    camera = CameraResponse(
        signal_source_sample_rate=15, frame_rate=30,
    )
    dummy_data = light(dummy_data)
    print("before camera:", dummy_data.shape)
    dummy_data = camera(dummy_data)
    print("After camera", dummy_data.shape)
    # print(dummy_data.shape)
    dummy_data.reshape((40, 30))
