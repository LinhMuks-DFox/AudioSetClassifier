import unittest
import sys

sys.path.append("..")
from src.FullSpectroAudioSet import FullSpectroAudioSet


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = FullSpectroAudioSet(
            r"../data/audio_set/AudioSet.json",
            "mix",
            41000,
            16000,
            512,
            256,
            128,
            True,
        )

    def test_length(self):
        print(len(self.dataset))


if __name__ == '__main__':
    unittest.main()
