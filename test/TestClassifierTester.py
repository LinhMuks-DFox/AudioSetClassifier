import unittest

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from MakeDummyModel import DummyModelForMNIST
from src.ClassifierTester import ClassifierTester


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.model = DummyModelForMNIST()
        self.model.load_state_dict(torch.load("dummy_model.pt"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        self.dataset = datasets.MNIST(root='data', train=True, transform=self.transform, download=True)
        self.dataset = torch.utils.data.Subset(self.dataset, range(10))
        self.classifier_tester: ClassifierTester = ClassifierTester(self.model, "cpu", multi_label=False)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=10, shuffle=True)
        self.classifier_tester.set_dataloader(self.test_loader, 10)

    def test_eval(self):
        for measure, score in self.classifier_tester.evaluate_model().items():
            print(f"{measure}: {score}")
        print(self.classifier_tester.y_true_)
        print(self.classifier_tester.y_predict_)


if __name__ == '__main__':
    unittest.main()
