from .AudioEncoder import AudioEncoder
from .AudioDecoder import AudioDecoder
import numpy as np
from typing import Dict, Any, Tuple


def make_auto_encoder_from_hyperparameter(data_shape, hyper: Dict[str, Any]) -> Tuple[AudioEncoder, AudioDecoder]:
    """
    Build the model from hyper
    :param data_shape:
    :param hyper:
    :return:
    """
    _encoder = AudioEncoder(data_shape,
                            out_feature=hyper["encoder_output_feature"],
                            conv_n_times=hyper["convolution_times"],
                            kernel_size=hyper["conv_kernel_size"],
                            out_channel=hyper["conv_output_channel"],
                            padding=hyper["conv_padding"],
                            stride=hyper["conv_stride"],
                            dilation=hyper["conv_dilation"])

    _decoder = AudioDecoder(
        linear_input_feature=hyper["encoder_output_feature"],
        shape_after_encoder_convolution=_encoder.shape_after_convolution_,
        encoder_input_shape=data_shape,
        conv_transpose_n_times=hyper["convolution_times"],
        kernel_size=np.flipud(hyper["conv_kernel_size"]),
        out_channels=np.flipud(hyper["conv_output_channel"]),
        padding=np.flipud(hyper["conv_padding"]),
        stride=np.flipud(hyper["conv_stride"]),
        dilation=np.flipud(hyper["conv_dilation"]),
        encoder_conv_layer_type=hyper["conv_type"]
    )
    return _encoder, _decoder
