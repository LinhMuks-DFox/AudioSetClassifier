import unittest

import torch
import torch.utils.data as tch_data

import MakeDummyModel
from src.MonoLabelClassifcationTester import MonoLabelClassificationTester


class OneHotClassificationTesterUnitTest(unittest.TestCase):

    def setUp(self):
        self.dummy_model = MakeDummyModel.DummyModelForMNIST()
        self.dummy_model.load_state_dict(torch.load("dummy_model.pt"))
        self.dummy_model.to("cuda:0")
        self.transformer = MakeDummyModel.get_transformer()
        self.dataset = MakeDummyModel.get_dataset()
        self.dataset, _, _ = tch_data.random_split(self.dataset, [0.2, 0.2, 0.6])
        self.dataloader = tch_data.DataLoader(self.dataset, batch_size=100)

        self.one_hot_tester = MonoLabelClassificationTester(self.dummy_model, torch.device("cuda:0"))
        self.one_hot_tester.set_dataloader(self.dataloader, 10)
        self.one_hot_tester.set_loss_function(
            MakeDummyModel.get_loss_function()
        )

    def test_predict(self):
        self.one_hot_tester.predict_all()
        print(self.one_hot_tester.y_predict_.shape)
        print(self.one_hot_tester.y_true_.shape)
        print(self.one_hot_tester.y_predict_)
        print(self.one_hot_tester.y_true_)

    def test_status_map(self):
        self.one_hot_tester.predict_all().calculate_all_metrics()
        print(self.one_hot_tester.status_map())


    def test_loss(self):
        print(self.one_hot_tester.loss_fn_)
        self.one_hot_tester.predict_all().calculate_all_metrics()
        print(self.one_hot_tester.loss_.shape)


if __name__ == '__main__':
    unittest.main()
