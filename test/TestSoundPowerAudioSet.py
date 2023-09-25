import unittest
import train_config
from src.SoundPowerAudioSet import SoundPowerAudioSet
import hyper_para


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_build_up()

    def test_build_up(self):
        self.dataset = SoundPowerAudioSet(
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE
        )

    def test_getitem(self):
        sample, label = self.dataset[0]
        print(sample.shape)


if __name__ == '__main__':
    unittest.main()
