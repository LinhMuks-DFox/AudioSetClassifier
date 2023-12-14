import torch.nn
import torch.utils.data
import torchaudio.transforms as tch_audio_trans

import lib.AudioSet.transform as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
from lib.AutoEncoder.AudioEncoder import AudioEncoder
from lib.AutoEncoder.AutoEncoderPrepare import make_auto_encoder_from_hyperparameter
from . import tags
from .LightPropaCamera import LightToCamera
from .util import label_digit2tensor, fix_length, blinky_data_normalize
from typing import Union


@tags.stable_api
class AutoEncodedAudioSet(torch.utils.data.Dataset):

    def __init__(self, auto_encoder_hypers,
                 encoder_model_path,
                 json_path: str,
                 n_class: int,
                 audio_sample_path: str,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 normalized: bool,
                 light_dis: float,
                 light_bias: Union[float, None],
                 light_std: float,
                 camera_source_sr: int,
                 camera_frame_rate: int,
                 camera_temperature: float = 0.1,
                 sample_seconds: int = 10,
                 output_size: tuple = (10, 80),
                 transform_device: torch.device = torch.device('cpu'),
                 one_hot_label: bool = True,
                 compile_model: bool = False
                 ):
        # region data fetch-transform
        self.transform_device_ = transform_device
        self.audio_fetcher_ = JsonBasedAudioSet(json_path, audio_sample_path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  win_length=win_length,
                                                                  normalized=normalized)
        self.amplitude_trans_ = tch_audio_trans.AmplitudeToDB()

        self.resampler_ = self.resampler_.to(self.transform_device_)
        self.amplitude_trans_ = self.amplitude_trans_.to(self.transform_device_)
        self.spectrogram_converter_ = self.spectrogram_converter_.to(self.transform_device_)

        # endregion

        # region properties
        self.sound_track_ = sound_track
        self.orig_freq_ = orig_freq
        self.new_freq_ = new_freq
        self.n_fft_ = n_fft
        self.hop_length_ = hop_length
        self.win_length_ = win_length
        self.normalized_ = normalized
        self.sample_seconds_ = sample_seconds
        self._split_i = self.sample_seconds_ * self.new_freq_ // 2
        self.output_size_ = output_size
        self.sample_length_ = self.sample_seconds_ * self.new_freq_

        self.n_class = n_class
        # endregion

        # region load encoder
        self.auto_encoder: AudioEncoder = make_auto_encoder_from_hyperparameter(self._data_shape_(),
                                                                                auto_encoder_hypers)[0]
        self.auto_encoder.load_state_dict(
            torch.load(encoder_model_path, map_location=self.transform_device_)
        )
        self.auto_encoder.to(self.transform_device_)
        self.one_hot_label_ = one_hot_label
        # endregion
        self.light_camera_ = LightToCamera(
            distance=light_dis,
            bias=light_bias,
            std=light_std,
            signal_source_sample_rate=camera_source_sr,
            frame_rate=camera_frame_rate,
            temperature=camera_temperature
        ).to(self.transform_device_)
        if compile_model:
            torch.compile(self.auto_encoder)

    def __len__(self):
        return len(self.audio_fetcher_)

    def __str__(self):
        return f"AutoEncodedAudioSet: length({len(self)}), using device: {self.transform_device_}\n"

    def _data_shape_(self):
        sample: torch.Tensor = self.audio_fetcher_[0][0]
        sample = sample.to(self.transform_device_)
        sample = self.track_selector_(sample)
        sample = self.resampler_(sample)
        sample = fix_length(sample, self.sample_length_)
        sample = sample[:, :80000]
        sample = self.spectrogram_converter_(sample)
        sample = self.amplitude_trans_(sample)
        return sample.shape

    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = label_digit2tensor(label_digits, self.n_class) if self.one_hot_label_ else torch.tensor(label_digits)
        sample: torch.Tensor = sample.to(self.transform_device_)
        track = self.track_selector_(sample)
        track = self.resampler_(track)
        track = fix_length(track, self.sample_length_)
        prev5s, post5s = track[:, :80000], track[:, 80000:]  # 10s * 16000Hz = 160000 samples
        prev5s_spe = self.spectrogram_converter_(prev5s)
        post5s_spe = self.spectrogram_converter_(post5s)
        prev5s_spe_db = self.amplitude_trans_(prev5s_spe)
        post5s_spe_db = self.amplitude_trans_(post5s_spe)
        prev5s_spe_db, post5s_spe_db = prev5s_spe_db.to(self.transform_device_), post5s_spe_db.to(
            self.transform_device_)
        pres5s_auto_encoded = self.auto_encoder(prev5s_spe_db)
        posts5s_auto_encoded = self.auto_encoder(post5s_spe_db)
        x = torch.hstack([pres5s_auto_encoded, posts5s_auto_encoded])
        x = blinky_data_normalize(x)  # normalize to [0, 1]
        x = x.reshape((4, -1))  # for 4 LED
        x = self.light_camera_(x)
        x = x.reshape((-1,))  # flatten
        x = x.reshape(self.output_size_)
        return x, label
