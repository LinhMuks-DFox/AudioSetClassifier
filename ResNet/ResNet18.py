"""
This ResNet18 implementation was designed to classify the Google AudioSet
Therefore the structure of this ResNet18 is different from the original ResNet18

* No maxpooling layer at the beginning
"""
import typing

import torch
import torch.nn as nn


class ResNet18(nn.Module):

    def __init__(self, label_num: int):
        super().__init__()
        self.label_num: int = label_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.ReLU()
        )

        self.conv2_x = [
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            )
        ]

        self.conv3_x = [
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            )
        ]

        self.conv4_x = [
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            )
        ]

        self.conv5_x = [
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            )
        ]

        self.avg_pool = nn.AvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential()

    @staticmethod
    def _residual_forward(x: torch.Tensor, conv: typing.List[nn.Sequential]):
        for layer in conv:
            x = layer(x) + x
        return x

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self._residual_forward(x, self.conv2_x)
        x = self._residual_forward(x, self.conv3_x)
        x = self._residual_forward(x, self.conv4_x)
        x = self._residual_forward(x, self.conv5_x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.mlp(x)
