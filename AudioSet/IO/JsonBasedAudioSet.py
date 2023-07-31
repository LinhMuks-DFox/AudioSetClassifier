import json
import os
from typing import Tuple, Optional, List

import torch
import torchaudio
from torch.utils.data import Dataset

SampleRateType = int
AudioSampleType = torch.Tensor
OntologyType = List[str]
LabelDigitsType = List[int]
LabelDisplayType = List[str]
JsonBasedAudioGetItemType = Tuple[AudioSampleType, SampleRateType, OntologyType, LabelDigitsType, LabelDisplayType]


class JsonBasedAudioSet(Dataset):
    def __init__(self,
                 json_file_path: str,
                 audio_sample_folder: Optional[str] = None) -> None:
        if not os.path.exists(json_file_path):
            raise IOError(f"Json file not found: <{json_file_path}>")
        if audio_sample_folder is not None and not os.path.exists(audio_sample_folder):
            raise IOError(f"Audio sample folder not found: <{audio_sample_folder}>")

        if audio_sample_folder is not None:
            self.json_name, self.audio_sample_folder = json_file_path, audio_sample_folder
        else:
            self.audio_sample_folder, self.json_name = os.path.split(json_file_path)
        self.json_obj = json.load(open(json_file_path, "r"))

    def __getitem__(self, idx: int) -> JsonBasedAudioGetItemType:
        sample_path = self.splice_audio_path(idx)
        if not os.path.exists(sample_path):
            raise IOError(f"Audio sample not found: <{sample_path}>")
        sample, sample_rate = torchaudio.load(sample_path, normalize=True)
        onto = self.json_obj[f"{idx}"]["onto"]
        label_digits = [int(i) for i in self.json_obj[f"{idx}"]["label_digits"]]
        label_display = self.json_obj[f"{idx}"]["label_display"]
        return sample, sample_rate, onto, label_digits, label_display

    def __len__(self):
        return len(self.json_obj)

    def splice_audio_path(self, idx: int) -> str:
        file_name = self.json_obj[f"{idx}"]["path"]
        file_path = os.path.join(self.audio_sample_folder, file_name)
        return file_path
