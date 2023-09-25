import typing

import torch
import sklearn.metrics as metrics
from . import tags


class ClassifierTester:

    def __init__(self, model: torch.nn.Module):
        self.model_ = model
        self.model_.eval()
        self.dataloader_ = None
        self.n_classes_ = None

        self.confusion_matrix_ = None
        self.accuracy_ = None
        self.precision_ = None
        self.recall_ = None
        self.f1_score_ = None

        self.y_predict_ = torch.empty(0)
        self.y_true_ = torch.empty(0)


    def set_dataloader(self, dataset, n_classes: int) -> "ClassifierTester":
        self.dataloader_ = dataset
        self.n_classes_ = n_classes
        return self

    @tags.untested
    def predict_all(self) -> "ClassifierTester":
        with torch.no_grad():
            for x, y in self.dataloader_:
                self.y_true_ = torch.hstack((self.y_true_, y))
                y_predict = self.model_(x)
                y_predict = torch.argmax(y_predict, dim=1)
                self.y_predict_ = torch.hstack((self.y_predict_, y_predict))
        return self

    @tags.untested
    def calculate_confusion_matrix(self) -> "ClassifierTester":
        self.confusion_matrix_ = metrics.confusion_matrix(self.y_true_, self.y_predict_)
        return self

    @tags.untested
    def calculate_accuracy(self) -> "ClassifierTester":
        self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_)
        return self

    @tags.untested
    def calculate_precision(self) -> "ClassifierTester":
        self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_, average='macro')
        return self

    @tags.untested
    def calculate_recall(self) -> "ClassifierTester":
        self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_, average='macro')
        return self

    @tags.untested
    def calculate_f1_score(self) -> "ClassifierTester":
        self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_, average='macro')
        return self

    def status_map(self) -> dict:
        return {
            "confusion_matrix": self.confusion_matrix_,
            "accuracy": self.accuracy_,
            "precision": self.precision_,
            "recall": self.recall_,
            "f1_score": self.f1_score_
        }

    def evaluate_model(self) -> typing.Dict[str, typing.Any]:
        self.predict_all()
        self.calculate_confusion_matrix()
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1_score()
        return self.status_map()
