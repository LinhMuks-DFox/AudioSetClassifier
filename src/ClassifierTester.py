import torch
import sklearn.metrics as metrics
import util


class ClassifierTester:

    def __init__(self, model: torch.nn.Module):
        self.model_ = model
        self.model_.eval()
        self.dataset_ = None
        self.n_classes_ = None

        self.confusion_matrix_ = None
        self.accuracy_ = None
        self.precision_ = None
        self.recall_ = None
        self.f1_score_ = None

        self.y_predict_ = None

    def set_dataset(self, dataset, n_classes: int):
        self.dataset_ = dataset
        self.n_classes_ = n_classes

    @util.untested()
    def fit(self):
        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataset_):
                y_predict = self.model_(x)
                y_predict = torch.argmax(y_predict, dim=1)
                if i == 0:
                    self.y_predict_ = y_predict
                else:
                    self.y_predict_ = torch.cat((self.y_predict_, y_predict), dim=0)

    # Create a confusion matrix
    def confusion_matrix(self):
        pass

    def accuracy(self):
        pass

    def precision(self):
        pass

    def recall(self):
        pass

    def f1_score(self):
        pass
