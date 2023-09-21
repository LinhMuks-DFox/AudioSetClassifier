from typing import *

import numpy as np
import torch
import torch.nn as nn

from . import utils

NDArray = np.ndarray
ValidPaddingMode = {
    "zeros",
    "reflect",
    "replicate",
    "circular"
}


class AudioEncoder(nn.Module):
    def __init__(self, input_dim: NDArray,
                 out_feature: int,
                 conv_n_times: int,
                 kernel_size: Optional[NDArray] = None,
                 out_channel: Optional[NDArray] = None,
                 output_length: Optional[NDArray] = None,
                 padding: Optional[NDArray] = None,
                 stride: Optional[NDArray] = None,
                 dilation: Optional[NDArray] = None,
                 conv_layer_type: Optional[Union[Type[nn.Conv1d], Type[nn.Conv2d], Type[nn.Conv3d]]] = None,
                 groups: Optional[List[int]] = None,
                 bias: Optional[List[bool]] = None,
                 active_function: Union[Type[nn.ReLU], Type[nn.Sigmoid], Type[nn.Softmax]] = None,
                 padding_mode: Optional[List[str]] = None
                 ):
        super().__init__()
        self.input_dim_: NDArray = input_dim
        self.out_feature_ = out_feature
        self.first_conv_layer_in_channel_ = self.input_dim_[0]
        self.conv_kernel_size_: NDArray = kernel_size \
            if kernel_size is not None else np.array([[3, 3] * self.conv_n_times_])
        self.conv_n_times_: int = conv_n_times if conv_n_times is not None else 2
        self.conv_layer_output_channels_: NDArray = out_channel \
            if out_channel is not None else np.array([1 for _ in range(self.conv_n_times_)])
        self.output_length_: NDArray = output_length \
            if output_length is not None else np.array([1, 8])
        self.conv_padding_: NDArray = padding \
            if padding is not None else np.array([[0, 0] for _ in range(self.conv_n_times_)])
        self.conv_stride_: NDArray = stride \
            if stride is not None else np.array([[1, 1] for _ in range(self.conv_n_times_)])
        self.conv_dilation_: NDArray = dilation \
            if dilation is not None else np.array([[1, 1] for _ in range(self.conv_n_times_)])

        self.conv_layer_type_ = conv_layer_type \
            if conv_layer_type is not None else \
            {
                1: nn.Conv1d,
                2: nn.Conv2d,
                3: nn.Conv3d
            }.get(len(input_dim) - 1)

        self.conv_groups_ = groups \
            if groups is not None else [1 for _ in range(self.conv_n_times_)]

        self.conv_bias_ = bias \
            if bias is not None else [False for _ in range(self.conv_n_times_)]

        self.active_function_type_ = active_function \
            if active_function is not None else nn.ReLU
        self.conv_padding_mode_: List[str] = padding_mode \
            if padding_mode is not None else ["zeros" for _ in range(self.conv_n_times_)]

        if not len(self.conv_kernel_size_) == self.conv_n_times_ == \
               len(self.conv_layer_output_channels_) == len(self.conv_padding_mode_):
            raise ValueError(
                f"size of kernel_size({len(self.conv_kernel_size_)}) and conv_n_times({self.conv_n_times_}) "
                f"and sizeof conv_layer_output_channels({len(self.conv_layer_output_channels_)}) and "
                f"padding mode length{len(self.conv_padding_mode_)} shall be equal.")

        self.shape_after_convolution_ = utils.shape_after_n_time_convolution(
            ndim_shape=self.input_dim_[1:],
            n_time=self.conv_n_times_,
            kernel_size=self.conv_kernel_size_,
            padding=self.conv_padding_,
            stride=self.conv_stride_,
            dilation=self.conv_dilation_
        )

        self.linear_layer_in_feature_ = int(np.prod(self.shape_after_convolution_))

        self.conv_layers_lists_ = [
            self.conv_layer_type_(in_channels=self.first_conv_layer_in_channel_,
                                  out_channels=self.conv_layer_output_channels_[0],
                                  kernel_size=self.conv_kernel_size_[0],
                                  padding=self.conv_padding_[0],
                                  stride=self.conv_stride_[0],
                                  dilation=self.conv_dilation_[0],
                                  groups=self.conv_groups_[0],
                                  bias=self.conv_bias_[0],
                                  padding_mode=self.conv_padding_mode_[0]
                                  ),
            self.active_function_type_()
        ]

        for i in range(1, self.conv_n_times_):
            self.conv_layers_lists_.append(
                self.conv_layer_type_(in_channels=self.conv_layer_output_channels_[i - 1],
                                      out_channels=self.conv_layer_output_channels_[i],
                                      kernel_size=self.conv_kernel_size_[i],
                                      padding=self.conv_padding_[i],
                                      stride=self.conv_stride_[i],
                                      dilation=self.conv_dilation_[i],
                                      groups=self.conv_groups_[i],
                                      bias=self.conv_bias_[i],
                                      padding_mode=self.conv_padding_mode_[i]))
            self.conv_layers_lists_.append(self.active_function_type_())

        self.flatten_layer_ = nn.Flatten()
        self.linear_layer_lists_ = [
            nn.Linear(in_features=self.linear_layer_in_feature_, out_features=self.out_feature_ * 2),
            self.active_function_type_(),
            nn.Linear(in_features=self.out_feature_ * 2, out_features=self.out_feature_)
        ]

        self.convolution_sequence_ = nn.Sequential(*self.conv_layers_lists_)
        self.linear_sequence_ = nn.Sequential(*self.linear_layer_lists_)

    def forward(self, sample: torch.Tensor):
        conv_ed = self.convolution_sequence_(sample)
        flatten = self.flatten_layer_(conv_ed)
        ret = self.linear_sequence_(flatten)
        return ret

    def __str__(self):
        # display all field of self, in a fancy format
        return f"{self.__class__.__name__}(\n\t" + \
            f"out_feature={self.out_feature_},\n\t" + \
            f"first_conv_layer_in_channel={self.first_conv_layer_in_channel_},\n\t" + \
            f"input_dim={self.input_dim_},\n\t" + \
            f"conv_n_times={self.conv_n_times_},\n\t" + \
            f"out_channel={self.conv_layer_output_channels_},\n\t" + \
            f"output_length={self.output_length_},\n\t" + \
            f"padding={self.conv_padding_},\n\t" + \
            f"stride={self.conv_stride_},\n\t" + \
            f"dilation={self.conv_dilation_},\n\t" + \
            f"conv_layer_type={self.conv_layer_type_},\n\t" + \
            f"groups={self.conv_groups_},\n\t" + \
            f"bias={self.conv_bias_},\n\t" + \
            f"active_function={self.active_function_type_},\n\t" + \
            f"padding_mode={self.conv_padding_mode_},\n\t" + \
            f"shape_after_convolution={self.shape_after_convolution_},\n\t" + \
            f"linear_layer_in_feature={self.linear_layer_in_feature_},\n\t" + \
            f"conv_layers_lists={self.conv_layers_lists_},\n\t" + \
            f"flatten_layer={self.flatten_layer_},\n\t" + \
            f"linear_layer_lists={self.linear_layer_lists_},\n\t" + \
            f"convolution_sequence={self.convolution_sequence_},\n\t" + \
            f"linear_sequence={self.linear_sequence_})"

    def __repr__(self):
        return f"{self.__class__.__name__} object at {hex(id(self))}"
