import torch
import torch.utils.data as tch_data
import torchaudio.transforms as tch_audio_trans
from typing import Union
import lib.AudioSet.transform as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
from .util import label_digit2tensor, fix_length, blinky_data_normalize
from .LightPropaCamera import LightToCamera


class SoundPowerAudioSet(tch_data.Dataset):
    def __init__(self, json_path: str,
                 audio_sample_path: str,
                 n_class: int,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 light_dis: float,
                 light_bias: Union[float, None],
                 light_std: float,
                 camera_source_sr: int,
                 camera_frame_rate: int,
                 camera_temperature: float = 0.1,
                 sound_power_data_count: int = 800,
                 output_size: tuple = (10, 80),
                 sample_seconds: int = 10,
                 transform_device: torch.device = torch.device('cpu'),
                 one_hot_label: bool = True,
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
            signal_source_sample_rate=camera_source_sr,
            frame_rate=camera_frame_rate,
            temperature=camera_temperature
        ).to(self.transform_device_)

    def __len__(self):
        return len(self.audio_fetcher_)

    def __str__(self):
        return f"SoundPowerAudioSet: {len(self)}, device: {self.transform_device_}, " \
               f"sample length: {self.sample_length_}, output size: {self.output_size_}"

    def __getitem__(self, index):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        sample: torch.Tensor = self.track_selector_(sample)
        sample = sample.to(self.transform_device_)
        sample = self.resampler_(sample)
        sample = fix_length(sample, self.sample_length_)
        # split sample to 800 chunks
        reshaped_sample = sample.view((self.sound_power_data_count_, -1))  # 16000Hz * 10s // 800 = 200
        sound_power = torch.sum(reshaped_sample ** 2, dim=1)
        sound_power = blinky_data_normalize(sound_power)
        # reshape them to 4 * N tensor, cause blinky has 4 LED.
        sound_power = sound_power.reshape((4, -1))
        # propagate value by light, and capture it with camera, (4 * 150) -> (4 * 300)
        sound_power = self.light_camera_(sound_power)
        # flatten the tensor, for now, we have 4 * 300 = 1200 floats
        sound_power = sound_power.reshape((-1,))
        # reshape to the output size
        sound_power = sound_power.reshape(self.output_size_)
        label = label_digit2tensor(label_digits, self.n_class) if self.one_hot_label_ else torch.tensor(label_digits)
        return torch.reshape(sound_power, self.output_size_), label
