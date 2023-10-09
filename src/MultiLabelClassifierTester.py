import typing

import sklearn.metrics as metrics
import torch

from . import tags


@tags.stable_api
class ClassifierTester:

    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 threshold: float = 0.5,
                 verbose: bool = True,
                 verbose_fn=print):
        self.model_ = model
        self.model_.eval()
        self.dataloader_ = None
        self.device_ = device

        self.n_classes_ = None

        self.confusion_matrix_ = None
        self.accuracy_ = None
        self.precision_ = None
        self.recall_ = None
        self.f1_score_ = None
        self.hamming_loss_ = None

        self.y_predict_ = None
        self.y_true_ = None
        self.y_predict_binary_ = None
        self.verbose_ = verbose
        self.verbose_fn_ = verbose_fn
        self.threshold_ = threshold

    def set_dataloader(self, dataloader, n_classes: int) -> "ClassifierTester":
        self.dataloader_ = dataloader
        self.n_classes_ = n_classes
        self.y_true_ = torch.zeros(1, n_classes, dtype=torch.int).to(self.device_)
        self.y_predict_ = torch.zeros(1, n_classes, dtype=torch.int).to(self.device_)
        return self

    def predict_all(self) -> "ClassifierTester":
        with torch.no_grad():
            for x, y in self.dataloader_:
                y = y.to(self.device_)
                x = x.to(self.device_)
                print(y.shape)
                y_predict = self.model_(x)
                self.y_true_ = torch.cat((self.y_true_, y))
                self.y_predict_ = torch.cat((self.y_predict_, y_predict))
        self.y_predict_ = self.y_predict_[1:]  # cut the first zero row
        self.y_true_ = self.y_true_[1:]  # cut the first zero row
        self.y_predict_ = self.y_predict_.detach().cpu().numpy()
        self.y_true_ = self.y_true_.to(torch.int).detach().cpu().numpy()
        self.y_predict_binary_ = (self.y_predict_ > self.threshold_).astype(int)
        return self

    def calculate_confusion_matrix(self) -> "ClassifierTester":
        self.confusion_matrix_ = metrics.multilabel_confusion_matrix(self.y_true_, self.y_predict_binary_)
        return self

    def calculate_accuracy(self) -> "ClassifierTester":
        self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_binary_)
        return self

    def calculate_precision(self) -> "ClassifierTester":
        self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_binary_, average='macro')
        return self

    def calculate_recall(self) -> "ClassifierTester":
        self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_binary_, average='macro')
        return self

    def calculate_f1_score(self) -> "ClassifierTester":
        self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_binary_, average='macro')
        return self

    def calculate_hamming_loss(self) -> "ClassifierTester":
        self.hamming_loss_ = metrics.hamming_loss(self.y_true_, self.y_predict_binary_)
        return self

    def status_map(self) -> dict:
        return {
            "confusion_matrix": self.confusion_matrix_,
            "accuracy": self.accuracy_,
            "precision": self.precision_,
            "recall": self.recall_,
            "f1_score": self.f1_score_,
            "hamming_loss": self.hamming_loss_
        }

    def evaluate_model(self) -> typing.Dict[str, typing.Any]:
        self.predict_all()
        self.calculate_confusion_matrix()
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1_score()
        self.calculate_hamming_loss()
        return self.status_map()

    def classification_report(self):
        return metrics.classification_report(self.y_true_, self.y_predict_binary_)
