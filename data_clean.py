import torchaudio.transforms
import torch.utils.data

import lib.AudioSet.transform
import train_config
import lib.AudioSet.IO as dataset_io

dataset = dataset_io.JsonBasedAudioSet(
    train_config.DATA_SET_PATH
)
resampler = torchaudio.transforms.Resample(44100, 16000)
sound_track_selector = lib.AudioSet.transform.SoundTrackSelector("mix")
cnt = 0
with open("error_file.txt", "w") as f:
    for i in range(len(dataset)):
        sample, sample_rate, onto, label_digits, label_display = dataset[i]
        sample = sound_track_selector(sample)
        sample = resampler(sample)
        if sample.shape != torch.Size([1, 160000]):
            print(dataset.splice_audio_path(i), sample.shape)
            cnt += 1
            f.write(f"file: {dataset.splice_audio_path(i)}, shape: {sample.shape}\n")
    f.write(f"Total non 16k*10 sample number:{cnt}\n")

