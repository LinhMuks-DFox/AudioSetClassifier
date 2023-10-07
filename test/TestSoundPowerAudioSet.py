import unittest

import hyper_para
import train_config
import train_prepare
from src.SoundPowerAudioSet import SoundPowerAudioSet


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_build_up()

    def test_build_up(self):
        self.dataset = SoundPowerAudioSet(
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE,
            transform_device=train_prepare.select_device(hyper_para.DATA_TRANSFORM_DEVICE)
        )

    def test_getitem(self):
        sample, label = self.dataset[0]
        print(sample.device)
        print(sample.shape)


if __name__ == '__main__':
    unittest.main()
