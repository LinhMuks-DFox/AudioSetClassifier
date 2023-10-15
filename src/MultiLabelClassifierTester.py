import typing
import warnings

import numpy as np
import sklearn.metrics as metrics
import torch
import tqdm


class MultiLabelClassifierTester:

    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 threshold: float,
                 use_sigmoid: bool = False):
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
        self.threshold_ = threshold
        self.use_sigmoid_ = use_sigmoid

        if self.use_sigmoid_ and self.threshold_ != .5:
            warnings.warn("When using sigmoid, decision threshold better be .5")

    def set_dataloader(self, dataloader, n_classes: int) -> "MultiLabelClassifierTester":
        self.dataloader_ = dataloader
        self.n_classes_ = n_classes
        self.y_true_ = torch.zeros(1, n_classes, dtype=torch.int).to(self.device_)
        self.y_predict_ = torch.zeros(1, n_classes, dtype=torch.float).to(self.device_)
        return self

    def predict_all(self) -> "MultiLabelClassifierTester":
        with torch.no_grad():
            for x, y in tqdm.tqdm(self.dataloader_):
                y = y.to(self.device_)
                x = x.to(self.device_)
                y_predict = self.model_(x)
                if self.use_sigmoid_:
                    y_predict = torch.sigmoid(y_predict)
                self.y_true_ = torch.cat((self.y_true_, y))
                self.y_predict_ = torch.cat((self.y_predict_, y_predict))
        self.y_predict_ = self.y_predict_[1:]  # cut the first zero row
        self.y_true_ = self.y_true_[1:]  # cut the first zero row
        self.y_predict_: np.ndarray = self.y_predict_.detach().cpu().numpy()
        self.y_true_: np.ndarray = self.y_true_.to(torch.int).detach().cpu().numpy()
        return self

    def make_binary_prediction(self, threshold: float = None) -> "MultiLabelClassifierTester":
        if self.y_predict_ is None or self.y_true_ is None:
            raise ValueError("call predict_all before make_binary_prediction")
        if self.use_sigmoid_:
            self.y_predict_binary_ = np.round(self.y_predict_)
            return self

        threshold = threshold if threshold is not None else self.threshold_
        self.y_predict_binary_ = self.y_predict_.copy()
        self.y_predict_binary_[self.y_predict_binary_ >= threshold] = 1
        self.y_predict_binary_[self.y_predict_binary_ < threshold] = 0
        self.y_predict_binary_ = self.y_predict_binary_.astype(np.int32)
        return self

    def calculate_confusion_matrix(self) -> "MultiLabelClassifierTester":
        self.confusion_matrix_ = metrics.multilabel_confusion_matrix(self.y_true_,
                                                                     self.y_predict_binary_)
        return self

    def calculate_accuracy(self) -> "MultiLabelClassifierTester":
        self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_binary_)
        return self

    def calculate_precision(self) -> "MultiLabelClassifierTester":
        self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_binary_,
                                                  average='macro', zero_division=np.nan)
        return self

    def calculate_recall(self) -> "MultiLabelClassifierTester":
        self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_binary_,
                                            average='macro', zero_division=np.nan)
        return self

    def calculate_f1_score(self) -> "MultiLabelClassifierTester":
        self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_binary_,
                                          average='macro', zero_division=np.nan)
        return self

    def calculate_hamming_loss(self) -> "MultiLabelClassifierTester":
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
        self.make_binary_prediction()
        self.calculate_all_metrics()
        return self.status_map()

    def calculate_all_metrics(self) -> "MultiLabelClassifierTester":
        self.calculate_confusion_matrix()
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1_score()
        self.calculate_hamming_loss()
        return self

    def classification_report(self):
        return metrics.classification_report(self.y_true_, self.y_predict_binary_, zero_division=np.nan)

    def __str__(self):
        return "MultiLabelClassifierTester at {}, using_sigmoid: {}, threshold({}): {}, device: {}\n".format(
            hex(id(self)), self.use_sigmoid_, "ignored" if self.use_sigmoid_ else "used", self.threshold_,
            self.device_
        )

    __repr__ = __str__
