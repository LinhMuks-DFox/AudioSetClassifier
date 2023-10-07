import torch
import torch.utils.data as data
import torchaudio.transforms as tch_audio_trans

import lib.AudioSet.transform as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
from . import tags
from .util import label_digit2tensor, fix_length


@tags.stable_api
class FullSpectroAudioSet(data.Dataset):
    def __init__(self, path: str,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 normalized: bool,
                 sample_seconds: int = 10,
                 transform_device: torch.device = torch.device("cpu")
                 ):
        super().__init__()
        self.transform_device_ = transform_device
        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq).to(self.transform_device_)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  win_length=win_length,
                                                                  normalized=normalized).to(self.transform_device_)
        self.sound_track_ = sound_track
        self.orig_freq_ = orig_freq
        self.new_freq_ = new_freq
        self.n_fft_ = n_fft
        self.hop_length_ = hop_length
        self.win_length_ = win_length
        self.normalized_ = normalized
        self.amplitude_trans = tch_audio_trans.AmplitudeToDB().to(self.transform_device_)
        self.sample_seconds_ = sample_seconds

    @tags.stable_api
    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        sample: torch.Tensor = sample.to(self.transform_device_)
        label = label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        resampled_track = self.resampler_(track)
        resampled_track = fix_length(resampled_track, self.new_freq_ * self.sample_seconds_)
        # fixed_track = self.length_fixer(resampled_track)
        spectrogram = self.spectrogram_converter_(resampled_track)
        db_spe = self.amplitude_trans(spectrogram)
        return db_spe, label

    @tags.stable_api
    def __len__(self):
        return len(self.audio_fetcher_)

    def __str__(self):
        return f"FullSpectroAudioSet(length={len(self)}), at {hex(id(self))}"