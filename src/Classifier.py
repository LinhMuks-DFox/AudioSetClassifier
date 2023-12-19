import torch
import torch.nn as nn
from collections import OrderedDict


class Classifier(nn.Module):

    def __init__(self, data_shape, n_class):
        super(Classifier, self).__init__()
        self.data_shape_ = data_shape
        if len(self.data_shape_) != 4:
            self.data_shape_ = torch.Size([1, 3, self.data_shape_[-2], self.data_shape_[-1]])
        self.n_class_ = n_class
        self.convolutions_ = nn.Sequential(OrderedDict([
            ("Conv1", nn.Conv2d(3, 16, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU1", nn.ReLU()),
            ("Conv2", nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU2", nn.ReLU()),
            ("Conv3", nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU3", nn.ReLU()),
            ("Conv4", nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU4", nn.ReLU()),
            ("Conv5", nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU5", nn.ReLU()),
            ("Conv6", nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU6", nn.ReLU()),
            ("Conv7", nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1))),
            ("ReLU7", nn.ReLU()),
            ("Conv8", nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))),
        ]))
        shape = self._shape_after_conv(self.data_shape_)
        self.flatten_ = nn.Flatten()
        # use multiplication(function in torch) to get the number of features
        in_features = torch.prod(torch.tensor(shape)).item()
        self.mlp = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(in_features, self.n_class_ * 32)),
            ("ReLU1", nn.ReLU()),
            ("Linear3", nn.Linear(self.n_class_ * 32, self.n_class_))
        ]))

    @torch.no_grad()
    def _shape_after_conv(self, shape):
        dummy_data = torch.randn(shape)
        return self.convolutions_(dummy_data).shape

    def forward(self, x):
        return self.mlp(self.flatten_(self.convolutions_(x)))


if __name__ == "__main__":
    model = Classifier((1, 3, 4, 300), 2)
    dummy_data = torch.randn((1, 3, 4, 300))
    print(model(dummy_data).shape)
    print(model)
