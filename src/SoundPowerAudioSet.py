import torch
import torch.utils.data as tch_data
import torchaudio.transforms as tch_audio_trans
from typing import Union
import lib.AudioSet.transform as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
from .util import label_digit2tensor, fix_length, blinky_data_normalize
from .LightPropaCamera import LightToCamera


class SoundPowerAudioSet(tch_data.Dataset):
    def __init__(self,
                 audio_sample_path: str,
                 camera_frame_rate: int,
                 camera_source_sr: int,
                 json_path: str,
                 light_bias: Union[float, None],
                 light_dis: float,
                 light_std: float,
                 n_class: int,
                 new_freq: int,
                 orig_freq: int,
                 sound_track: str,
                 camera_temperature: float = 0.1,
                 one_hot_label: bool = True,
                 output_size: tuple = (10, 80),
                 sample_seconds: int = 10,
                 sound_power_data_count: int = 800,
                 transform_device: torch.device = torch.device('cpu'),
                 ):
        self.transform_device_ = transform_device
        self.audio_fetcher_ = JsonBasedAudioSet(json_path, audio_sample_path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = (tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
                           .to(self.transform_device_))
        self.new_freq_ = new_freq
        self.sample_seconds_ = sample_seconds
        self.sound_power_data_count_ = sound_power_data_count
        self.output_size_ = output_size
        self.sample_length_ = self.sample_seconds_ * self.new_freq_
        self.n_class = n_class
        self.one_hot_label_ = one_hot_label
        self.light_camera_ = LightToCamera(
            distance=light_dis,
            bias=light_bias,
            std=light_std,
            signal_source_sample_rate=self.new_freq_ // 4,
            frame_rate=camera_frame_rate,
            temperature=camera_temperature
        ).to(self.transform_device_)

    def __len__(self):
        return len(self.audio_fetcher_)

    def __str__(self):
        return f"SoundPowerAudioSet: {len(self)}, device: {self.transform_device_}, " \
               f"sample length: {self.sample_length_}, output size: {self.output_size_}"

    @torch.no_grad()
    def __getitem__(self, index):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        sample: torch.Tensor = self.track_selector_(sample)
        sample = sample.to(self.transform_device_)
        sample = self.resampler_(sample)
        sample = fix_length(sample, self.sample_length_)  # 16k * 10s
        sound_power = sample ** 2
        sound_power = sound_power.reshape((4, -1))
        sound_power = self.light_camera_(sound_power)
        label = label_digit2tensor(label_digits, self.n_class) if self.one_hot_label_ else torch.tensor(label_digits)
        return torch.reshape(sound_power, self.output_size_), label
