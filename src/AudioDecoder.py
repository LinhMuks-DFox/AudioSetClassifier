from typing import *

import numpy as np
import torch
import torch.nn as nn

NDArray = np.ndarray

ValidPaddingMode = {
    "zeros",
    "reflect",
    "replicate",
    "circular"
}
EncoderConvLayerTypeToConvTransposeLayerType = {
    "Conv1d": nn.ConvTranspose1d,
    "Conv2d": nn.ConvTranspose2d,
    "Conv3d": nn.ConvTranspose3d,
    nn.Conv2d: nn.ConvTranspose2d,
    nn.Conv1d: nn.ConvTranspose1d,
    nn.Conv3d: nn.ConvTranspose3d
}


class AudioDecoder(torch.nn.Module):
    def __init__(self,
                 linear_input_feature: int,
                 shape_after_encoder_convolution: NDArray,
                 conv_transpose_n_times: int,
                 encoder_input_shape: NDArray,
                 kernel_size: Optional[NDArray] = None,
                 out_channels: Optional[NDArray] = None,
                 padding: Optional[NDArray] = None,
                 stride: Optional[NDArray] = None,
                 dilation: Optional[NDArray] = None,
                 encoder_conv_layer_type: Optional[str] = None,
                 groups: Optional[List[int]] = None,
                 bias: Optional[List[bool]] = None,
                 active_function: Union[Type[nn.ReLU], Type[nn.Sigmoid], Type[nn.Softmax]] = None,
                 padding_modes: Optional[List[str]] = None
                 ):
        super().__init__()
        self.linear_input_feature_: int = linear_input_feature
        self.shape_after_encoder_convolution_: NDArray = shape_after_encoder_convolution
        self.conv_transpose_n_times_: int = conv_transpose_n_times
        self.encoder_input_shape_ = encoder_input_shape
        self.conv_kernel_size_: NDArray = kernel_size \
            if kernel_size is not None else \
            np.array([[3, 3] for _ in range(self.conv_transpose_n_times_)])
        self.conv_output_channels_: NDArray = out_channels \
            if out_channels is not None else \
            np.array([1 for _ in range(self.conv_transpose_n_times_)])
        self.conv_padding_: NDArray = padding \
            if padding is not None else \
            np.array([[0, 0] for _ in range(self.conv_transpose_n_times_)])
        self.conv_stride_: NDArray = stride \
            if stride is not None else \
            np.array([[1, 1] for _ in range(self.conv_transpose_n_times_)])
        self.conv_dilation_: NDArray = dilation \
            if dilation is not None else \
            np.array([[1, 1] for _ in range(self.conv_transpose_n_times_)])
        self.convolution_transpose_layer_type_ = EncoderConvLayerTypeToConvTransposeLayerType[encoder_conv_layer_type] \
            if encoder_conv_layer_type is not None else nn.Conv2d
        self.conv_groups_: List[int] = groups \
            if groups is not None else \
            [1 for _ in range(self.conv_transpose_n_times_)]
        self.conv_bias_: List[bool] = bias \
            if bias is not None else \
            [False for _ in range(self.conv_transpose_n_times_)]
        self.conv_padding_modes_: List[str] = padding_modes \
            if padding_modes is not None else \
            ["zeros" for _ in range(self.conv_transpose_n_times_)]
        self.active_function_type_ = active_function \
            if active_function is not None else nn.ReLU

        self.linear_layers_output_features_: int = int(np.prod(self.shape_after_encoder_convolution_))
        self.linear_layer_list_ = [
            nn.Linear(self.linear_input_feature_, self.linear_input_feature_ * 2),
            self.active_function_type_(),
            nn.Linear(self.linear_input_feature_ * 2, self.linear_layers_output_features_)
        ]
        # Considering [batch size]
        self.unflatten = nn.Unflatten(1, tuple([int(self.encoder_input_shape_[0])] +
                                               [int(i) for i in self.shape_after_encoder_convolution_]))

        self.convolution_transpose_layer_lists_ = [
            self.convolution_transpose_layer_type_(in_channels=1,
                                                   out_channels=self.conv_output_channels_[0],
                                                   kernel_size=self.conv_kernel_size_[0],
                                                   padding=self.conv_padding_[0],
                                                   stride=self.conv_stride_[0],
                                                   dilation=self.conv_dilation_[0],
                                                   groups=self.conv_groups_[0],
                                                   bias=self.conv_bias_[0],
                                                   padding_mode=self.conv_padding_modes_[0]
                                                   ),
            self.active_function_type_()
        ]

        for i in range(1, self.conv_transpose_n_times_):
            self.convolution_transpose_layer_lists_.append(
                self.convolution_transpose_layer_type_(in_channels=self.conv_output_channels_[i - 1],
                                                       out_channels=self.conv_output_channels_[i],
                                                       kernel_size=self.conv_kernel_size_[i],
                                                       padding=self.conv_padding_[i],
                                                       stride=self.conv_stride_[i],
                                                       dilation=self.conv_dilation_[i],
                                                       groups=self.conv_groups_[i],
                                                       bias=self.conv_bias_[i],
                                                       padding_mode=self.conv_padding_modes_[i]))
            if i != self.conv_transpose_n_times_ - 1:  # Last layer does not need active function
                self.convolution_transpose_layer_lists_.append(self.active_function_type_())

        self.linear_sequence_ = nn.Sequential(*self.linear_layer_list_)
        self.convolution_transpose_sequence_ = nn.Sequential(*self.convolution_transpose_layer_lists_)

    def forward(self, sample: torch.Tensor):
        linear_output = self.linear_sequence_(sample)
        unflatten_output = self.unflatten(linear_output)
        convolution_transpose_output = self.convolution_transpose_sequence_(unflatten_output)
        return convolution_transpose_output

    def __str__(self):
        # Display ALL self Fields, In a fancy, as fancy as possible, way
        return "Decoder: \n" + \
            "Linear Input Feature: " + str(self.linear_input_feature_) + "\n" + \
            "Shape After Encoder Convolution: " + str(self.shape_after_encoder_convolution_) + "\n" + \
            "Convolution Transpose N Times: " + str(self.conv_transpose_n_times_) + "\n" + \
            "Convolution Kernel Size: " + str(self.conv_kernel_size_) + "\n" + \
            "Convolution Output Channels: " + str(self.conv_output_channels_) + "\n" + \
            "Convolution Padding: " + str(self.conv_padding_) + "\n" + \
            "Convolution Stride: " + str(self.conv_stride_) + "\n" + \
            "Convolution Dilation: " + str(self.conv_dilation_) + "\n" + \
            "Convolution Transpose Layer Type: " + self.convolution_transpose_layer_type_.__name__ + "\n" + \
            "Convolution Groups: " + str(self.conv_groups_) + "\n" + \
            "Convolution Bias: " + str(self.conv_bias_) + "\n" + \
            "Convolution Padding Modes: " + str(self.conv_padding_modes_) + "\n" + \
            "Active Function Type: " + self.active_function_type_.__name__ + "\n" + \
            "Linear Layers Output Features: " + str(self.linear_layers_output_features_) + "\n" + \
            "Linear Layer List: " + str(self.linear_layer_list_) + "\n" + \
            "Convolution Transpose Layer Lists: " + str(self.convolution_transpose_layer_lists_) + "\n" + \
            "Linear Sequence: " + str(self.linear_sequence_) + "\n" + \
            "Convolution Transpose Square Sequence: " + str(self.convolution_transpose_sequence_) + "\n"

    def __repr__(self):
        return f"{self.__class__.__name__} object at {hex(id(self))}"
