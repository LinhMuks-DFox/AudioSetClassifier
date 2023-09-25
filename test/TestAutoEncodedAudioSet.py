import unittest

import hyper_para
import train_config
from src.AutoEncodedAudioSet import AutoEncodedAudioSet


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = None
        self.test_build_up()

    def test_build_up(self) -> None:
        self.dataset = AutoEncodedAudioSet(
            auto_encoder_hypers=hyper_para.AUTO_ENCODER_MODEL,
            encoder_model_path=r"../pre_trained_encoder/2023-08-04-09-49-03/encoder.pth",
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE
        )

    def test_getitem(self):
        sample, label = self.dataset[0]
        print(sample.shape)


if __name__ == '__main__':
    unittest.main()
