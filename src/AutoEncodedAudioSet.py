import torch.nn
import torch.utils.data
from lib.AutoEncoder import *
from lib.AudioSet.IO import JsonBasedAudioSet
import lib.AudioSet.transform as sc_transforms
import torchaudio.transforms as tch_audio_trans
from .util import label_digit2tensor
from . import tags


@tags.untested
class AutoEncodedAudioSet(torch.utils.data.Dataset):
    def __init__(self, auto_encoder_hypers,
                 encoder_model_path,
                 path: str,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 normalized: bool,
                 sample_seconds: int = 10
                 ):
        self.auto_encoder: torch.nn.Module = make_auto_encoder_from_hyperparameter(auto_encoder_hypers)
        self.auto_encoder.load_state_dict(torch.load(encoder_model_path))

        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  win_length=win_length,
                                                                  normalized=normalized)

        self.sound_track_ = sound_track
        self.orig_freq_ = orig_freq
        self.new_freq_ = new_freq
        self.n_fft_ = n_fft
        self.hop_length_ = hop_length
        self.win_length_ = win_length
        self.normalized_ = normalized
        self.sample_seconds_ = sample_seconds
        self.amplitude_trans_ = tch_audio_trans.AmplitudeToDB()
        self._split_i = self.sample_seconds_ * self.new_freq_ // 2

    def __len__(self):
        return len(self.audio_fetcher_)

    @tags.untested
    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        track = self.resampler_(track)

        # TODO: split sound to 5 seconds
        _i = self.sample_seconds_ * self.new_freq_
        prev5s, post5s = (track[0:self._split_i // 2],
                          track[self._split_i // 2:])  # 10s * 16000Hz = 160000 samples

        prev5s_spe = self.spectrogram_converter_(prev5s)
        post5s_spe = self.spectrogram_converter_(post5s)
        x = self.spectrogram_converter_(post5s)
        #
        prev5s_spe_db = self.amplitude_trans(prev5s_spe)
        post5s_spe_db = self.amplitude_trans(post5s_spe)

        pres5s_auto_encoded = self.auto_encoder(prev5s_spe_db)
        posts5s_auto_encoded = self.auto_encoder(post5s_spe_db)

        # concat
        x = torch.cat([pres5s_auto_encoded, posts5s_auto_encoded], dim=0)
        x.reshape(10, 80)
        return x, label
