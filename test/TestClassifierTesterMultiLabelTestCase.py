import unittest

import torch
import torch.utils.data as tch_data

import hyper_para
import train_prepare
from src.MultiLabelClassifierTester import ClassifierTester


class TestClassifierTesterMultiLabelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_pt_path = r"../pth_bin/2023-10-08-17-58-50/ideal/checkpoint0.pt"
        self.model = train_prepare.make_classifier()
        self.model.load_state_dict(torch.load(self.model_pt_path))
        self.model = self.model.to(torch.device("cuda:1"))

        self.dataset = train_prepare.make_dataset("../data/audio_set/AudioSet.json")
        self.dataset = tch_data.Subset(self.dataset, list(range(10)))
        self.dataloader = tch_data.DataLoader(self.dataset, shuffle=False, batch_size=2)

        self.tester: ClassifierTester = ClassifierTester(self.model, torch.device("cuda:1"))
        self.tester.set_dataloader(self.dataloader, hyper_para.CLASS_CNT)

    def test_predict_all(self):
        self.tester.predict_all()
        print(self.tester.y_predict_.shape)
        print(self.tester.y_true_.shape)
        self.assertTrue(self.tester.y_true_.shape == self.tester.y_predict_.shape)


if __name__ == '__main__':
    unittest.main()
