import torch
import torch.utils.data as tch_data
import torchaudio.transforms as tch_audio_trans

import lib.AudioSet.transform as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
from . import tags
from .util import label_digit2tensor, fix_length



class SoundPowerAudioSet(tch_data.Dataset):
    def __init__(self, path: str,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 output_size: tuple = (10, 80),
                 sample_seconds: int = 10,
                 transform_device: torch.device = torch.device('cpu')
                 ):
        self.transform_device_ = transform_device
        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = (tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
                           .to(self.transform_device_))
        self.new_freq_ = new_freq
        self.sample_seconds_ = sample_seconds
        self.reshape_size_ = (800, 200)
        self.output_size_ = output_size
        self.sample_length_ = self.sample_seconds_ * self.new_freq_

    
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
        reshaped_sample = sample.reshape(*self.reshape_size_)  # 16000Hz * 10s // 800 = 200
        sound_power = torch.sum(reshaped_sample ** 2, dim=1)
        label = label_digit2tensor(label_digits)
        return sound_power.reshape(self.output_size_), label
