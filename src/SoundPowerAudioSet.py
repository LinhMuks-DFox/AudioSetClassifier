import tags
import torch
import torch.utils.data as tch_data
from lib.AudioSet.IO import JsonBasedAudioSet
from lib import AudioSet as sc_transforms
import torchaudio.transforms as tch_audio_trans
from util import label_digit2tensor


@tags.untested
class SoundPowerAudioSet(tch_data.Dataset):
    def __init__(self, path: str,
                 sound_track: str,
                 orig_freq: int,
                 new_freq: int,
                 sample_seconds: int = 10
                 ):
        self.audio_fetcher_ = JsonBasedAudioSet(path)
        self.track_selector_ = sc_transforms.SoundTrackSelector(sound_track)
        self.resampler_ = tch_audio_trans.Resample(orig_freq=orig_freq, new_freq=new_freq)
        self.reshape_size_ = (800, 200)

    def __len__(self):
        return len(self.audio_fetcher_)

    @tags.untested
    def __getitem__(self, index):
        sample, sample_rate, onto, label_digits, label_display = self.audio_fetcher_[index]
        # split sample to 800 chunks
        reshaped_sample = sample.reshape(*self.reshape_size_)  # 16000Hz * 10s // 800 = 200
        sound_power = torch.sum(reshaped_sample ** 2, dim=1) \
            .reshape(80, 10)
        label = label_digit2tensor(label_digits)

        return sound_power, label
