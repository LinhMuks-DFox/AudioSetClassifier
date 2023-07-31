import typing

import torch
import torch.utils.data as data
import torchaudio.transforms as tch_audio_trans

import AudioSet.transform.transforms as sc_transforms
from AudioSet.IO.JsonBasedAudioSet import JsonBasedAudioSet


class AudioSetForResNet18Classifier(data.Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector("mix")
        self.resampler_ = tch_audio_trans.Resample(orig_freq=44100, new_freq=16000)
        self.length_fixer = sc_transforms.TimeSequenceLengthFixer(5, 16000)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=512,
                                                                  hop_length=256,
                                                                  win_length=512,
                                                                  normalized=True)
        self.amplitude_trans = tch_audio_trans.AmplitudeToDB()

    @staticmethod
    def label_digit2tensor(label: typing.List[int]) -> torch.Tensor:
        t = torch.zeros(527)
        idx = torch.tensor(label)
        t[idx] = 1
        return t

    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = self.label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        resampled_track = self.resampler_(track)
        fixed_track = self.length_fixer(resampled_track)
        spectrogram = self.spectrogram_converter_(fixed_track)
        db_spe = self.amplitude_trans(spectrogram)
        return db_spe, label

    def __len__(self):
        return len(self.audio_fetcher_)
