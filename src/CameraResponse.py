import torch
import torch.nn as nn
import torchaudio


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
            resampling_method='sinc_interpolation'
        )

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x = self.resample(x)
        # x = deepy.nn.functional.softstaircase(x, self.levels, self.temperature)
        return x
