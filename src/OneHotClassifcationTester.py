import typing

import numpy as np
import sklearn.metrics as metrics
import torch


class OneHotClassificationTester:

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 ):
        self.model_ = model
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

    def set_dataloader(self, dataloader, n_class: int) -> "OneHotClassificationTester":
        self.dataloader_ = dataloader
        self.n_classes_ = n_class
        self.y_predict_ = torch.zeros(0, dtype=torch.int32).to(self.device_)
        self.y_true_ = torch.zeros(0, dtype=torch.int32).to(self.device_)
        return self

    def predict_all(self) -> "OneHotClassificationTester":
        if self.dataloader_ is None or self.y_predict_ is None or self.y_true_ is None:
            raise ValueError("dataloader, y_predict, y_true is None, use set_dataloader() before calling predict_all")
        self.model_.eval()
        self.model_.to(self.device_)
        with torch.no_grad():
            data: torch.Tensor
            label: torch.Tensor
            for data, label in self.dataloader_:
                data = data.to(self.device_)
                label = label.to(self.device_)
                predicted_y = torch.argmax(self.model_(data), dim=1)
                # if label is one-hot, convert it to int
                if len(label.shape) > 1:
                    label = torch.argmax(label, dim=1)
                self.y_true_ = torch.cat([self.y_true_, label])
                self.y_predict_ = torch.cat([self.y_predict_, predicted_y])

        self.y_true_: np.ndarray = self.y_true_.detach().cpu().numpy()
        self.y_predict_: np.ndarray = self.y_predict_.detach().cpu().numpy()
        return self

    def calculate_confusion_matrix(self) -> "OneHotClassificationTester":
        self.confusion_matrix_ = metrics.confusion_matrix(self.y_true_, self.y_predict_)
        return self

    def calculate_accuracy(self, ) -> "OneHotClassificationTester":
        self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_)
        return self

    def calculate_precision(self, ) -> "OneHotClassificationTester":
        self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_, average="macro", zero_division=np.nan)
        return self

    def calculate_recall(self, ) -> "OneHotClassificationTester":
        self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_, average="macro", zero_division=np.nan)
        return self

    def calculate_f1_score(self, ) -> "OneHotClassificationTester":
        self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_, average="macro", zero_division=np.nan)
        return self

    def calculate_hamming_loss(self, ) -> "OneHotClassificationTester":
        self.hamming_loss_ = metrics.hamming_loss(self.y_true_, self.y_predict_)
        return self

    def status_map(self) -> typing.Dict:
        if not all([
            self.f1_score_ is not None,
            self.accuracy_ is not None,
            self.precision_ is not None,
            self.recall_ is not None,
            self.hamming_loss_ is not None,
            self.confusion_matrix_ is not None,
        ]):
            raise ValueError("None metrics exist, use calculate_all_metrics() before calling status_map")
        return {
            "f1_score_": self.f1_score_,
            "accuracy": self.accuracy_,
            "precision": self.precision_,
            "recall": self.recall_,
            "hamming_loss": self.hamming_loss_,
            "confusion_matrix": self.confusion_matrix_,

        }

    def evaluate_model(self):
        self.predict_all()
        self.calculate_all_metrics()
        return self.status_map()

    def calculate_all_metrics(self) -> "OneHotClassificationTester":
        if self.y_true_ is None or self.y_predict_ is None:
            raise ValueError("y_true, y_predict is None, use predict_all() before calling calculate_all_metrics")
        if isinstance(self.y_true_, torch.Tensor) or isinstance(self.y_predict_, torch.Tensor):
            raise ValueError("y_true, y_predict is None, use predict_all() before calling calculate_all_metrics")
        self.calculate_recall()
        self.calculate_f1_score()
        self.calculate_precision()
        self.calculate_accuracy()
        self.calculate_hamming_loss()
        self.calculate_confusion_matrix()
        return self

    def classification_report(self, ):
        return metrics.classification_report(self.y_true_, self.y_predict_, zero_division=np.nan)
