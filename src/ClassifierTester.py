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

        self.y_predict_ = None

    def set_dataloader(self, dataset, n_classes: int) -> "ClassifierTester":
        self.dataloader_ = dataset
        self.n_classes_ = n_classes
        return self

    @tags.untested
    def eval(self) -> "ClassifierTester":
        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataloader_):
                y_predict = self.model_(x)
                y_predict = torch.argmax(y_predict, dim=1)
                if i == 0:
                    self.y_predict_ = y_predict
                else:
                    self.y_predict_ = torch.cat((self.y_predict_, y_predict), dim=0)
        return self

    @tags.untested
    def calculate_confusion_matrix(self) -> "ClassifierTester":
        self.confusion_matrix_ = metrics.confusion_matrix(self.dataloader_.dataset.y, self.y_predict_)
        return self

    @tags.untested
    def calculate_accuracy(self) -> "ClassifierTester":
        self.accuracy_ = metrics.accuracy_score(self.dataloader_.dataset.y, self.y_predict_)

    @tags.untested
    def calculate_precision(self) -> "ClassifierTester":
        self.precision_ = metrics.precision_score(self.dataloader_.dataset.y, self.y_predict_, average='macro')
        return self

    @tags.untested
    def calculate_recall(self) -> "ClassifierTester":
        self.recall_ = metrics.recall_score(self.dataloader_.dataset.y, self.y_predict_, average='macro')
        return self

    @tags.untested
    def calculate_f1_score(self) -> "ClassifierTester":
        self.f1_score_ = metrics.f1_score(self.dataloader_.dataset.y, self.y_predict_, average='macro')
        return self

    def status_map(self) -> dict:
        return {
            "confusion_matrix": self.confusion_matrix_,
            "accuracy": self.accuracy_,
            "precision": self.precision_,
            "recall": self.recall_,
            "f1_score": self.f1_score_
        }
