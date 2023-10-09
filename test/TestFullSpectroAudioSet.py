import unittest

import torch

import hyper_para
import train_config
import train_prepare
from src.FullSpectroAudioSet import FullSpectroAudioSet


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_build_up()
        self.data_on_device = (torch.empty(0)
                               .to(train_prepare.select_device(hyper_para.DATA_TRANSFORM_DEVICE)))

    def test_build_up(self):
        self.dataset = FullSpectroAudioSet(
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
            transform_device=train_prepare.select_device(hyper_para.DATA_TRANSFORM_DEVICE)
        )

    def test_getitem(self):
        self.sample0, self.label0 = self.dataset[0]
        print(self.sample0.shape)

    def test_device(self):
        self.test_getitem()
        print(self.sample0.device)
        print(self.data_on_device.device)
        self.assertTrue(self.sample0.device == self.data_on_device.device)


if __name__ == '__main__':
    unittest.main()
