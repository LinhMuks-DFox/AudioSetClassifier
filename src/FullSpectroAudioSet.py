import torch.utils.data as data
import torchaudio.transforms as tch_audio_trans

from lib import AudioSet as sc_transforms
from lib.AudioSet.IO import JsonBasedAudioSet
import tags
from util import label_digit2tensor


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
                 ):
        super().__init__()
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
        self.amplitude_trans = tch_audio_trans.AmplitudeToDB()

    def __getitem__(self, index: int):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        label = label_digit2tensor(label_digits)
        track = self.track_selector_(sample)
        resampled_track = self.resampler_(track)
        # fixed_track = self.length_fixer(resampled_track)
        spectrogram = self.spectrogram_converter_(resampled_track)
        db_spe = self.amplitude_trans(spectrogram)
        return db_spe, label

    def __len__(self):
        return len(self.audio_fetcher_)
