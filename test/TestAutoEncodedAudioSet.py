import unittest

import torch
import hyper_para
import train_config
import train_prepare
from src.AutoEncodedAudioSet import AutoEncodedAudioSet


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = None
        self.test_build_up()
        self.data_on_device = (torch.empty(0)
                               .to(train_prepare.select_device(hyper_para.DATA_TRANSFORM_DEVICE)))

    def test_build_up(self) -> None:
        self.dataset = AutoEncodedAudioSet(
            auto_encoder_hypers=hyper_para.AUTO_ENCODER_MODEL,
            encoder_model_path=r"../pre_trained_encoder/2023-08-04-09-49-03/encoder.pth",
            path=r"../data/audio_set/AudioSet.json",
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE,
            transform_device=hyper_para.DATA_TRANSFORM_DEVICE,
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
