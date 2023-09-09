import torch.nn
import torch.utils.data
from lib.AutoEncoder import *
from lib.AudioSet.IO import JsonBasedAudioSet
import lib.AudioSet.transform as sc_transforms
import torchaudio.transforms as tch_audio_trans
from src.util import label_digit2tensor
import tags


@tags.untested
class AutoEncodedAudioSet(torch.utils.data.Dataset):
    def __init__(self, auto_encoder_hypers, encoder_model_path, path: str):
        self.auto_encoder: torch.nn.Module = make_auto_encoder_from_hyperparameter(auto_encoder_hypers)
        self.auto_encoder.load_state_dict(torch.load(encoder_model_path))

        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector("mix")
        self.resampler_ = tch_audio_trans.Resample(orig_freq=44100, new_freq=16000)
        self.spectrogram_converter_ = tch_audio_trans.Spectrogram(n_fft=512,
                                                                  hop_length=256,
                                                                  win_length=512,
                                                                  normalized=True)
        self.amplitude_trans = tch_audio_trans.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_fetcher_)

    @tags.unfinished_api
    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        track = self.resampler_(track)

        # TODO: split sound to 5 seconds
        prev5s, post5s = track[0], track[0]

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
