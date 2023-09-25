import torch.nn
import torch.utils.data
from lib.AutoEncoder import *
from lib.AudioSet.IO import JsonBasedAudioSet
import lib.AudioSet.transform as sc_transforms
import torchaudio.transforms as tch_audio_trans
from .util import label_digit2tensor
from . import tags
from lib.AutoEncoder.AutoEncoderPrepare import make_auto_encoder_from_hyperparameter
from lib.AutoEncoder.AudioEncoder import AudioEncoder


@tags.stable_api
class AutoEncodedAudioSet(torch.utils.data.Dataset):

    @tags.stable_api
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
        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  win_length=win_length,
                                                                  normalized=normalized)
        self.amplitude_trans_ = tch_audio_trans.AmplitudeToDB()
        # self.time_zone_selector_ = sc_transforms.TimeZoneSelector(sample_seconds)
        self.sound_track_ = sound_track
        self.orig_freq_ = orig_freq
        self.new_freq_ = new_freq
        self.n_fft_ = n_fft
        self.hop_length_ = hop_length
        self.win_length_ = win_length
        self.normalized_ = normalized
        self.sample_seconds_ = sample_seconds
        self._split_i = self.sample_seconds_ * self.new_freq_ // 2

        self.auto_encoder: AudioEncoder = make_auto_encoder_from_hyperparameter(self._data_shape_(),
                                                                                auto_encoder_hypers)[0]
        self.auto_encoder.load_state_dict(torch.load(encoder_model_path))

    @tags.stable_api
    def __len__(self):
        return len(self.audio_fetcher_)

    @tags.stable_api
    def _data_shape_(self):
        sample = self.audio_fetcher_[0][0]
        sample = self.track_selector_(sample)
        sample = self.resampler_(sample)
        sample = sample[:, :80000]
        sample = self.spectrogram_converter_(sample)
        sample = self.amplitude_trans_(sample)
        return sample.shape

    @tags.stable_api
    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        track = self.resampler_(track)

        prev5s, post5s = track[:, :80000], track[:, 80000:]  # 10s * 16000Hz = 160000 samples

        prev5s_spe = self.spectrogram_converter_(prev5s)
        post5s_spe = self.spectrogram_converter_(post5s)

        prev5s_spe_db = self.amplitude_trans_(prev5s_spe)
        post5s_spe_db = self.amplitude_trans_(post5s_spe)

        pres5s_auto_encoded = self.auto_encoder(prev5s_spe_db)
        posts5s_auto_encoded = self.auto_encoder(post5s_spe_db)

        x = torch.hstack([pres5s_auto_encoded, posts5s_auto_encoded])
        # x.reshape(80, 10)
        return x.reshape(80, 10), label
