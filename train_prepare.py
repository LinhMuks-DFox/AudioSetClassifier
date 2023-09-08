import torch.nn
import torch.utils.data as tch_data
import torchvision
from typing import Tuple, Dict, Any
from src.AudioEncoder import AudioEncoder
from src.AudioDecoder import AudioDecoder
import numpy as np
import hyper_para
from src.AutoEncodedAudioSet import AutoEncodedAudioSet
from src.FullSpectroAudioSet import FullSpectroAudioSet
from src.SoundPowerAudioSet import SoundPowerAudioSet
import platform


def make_model():
    _CASE = {
        "RES18": torchvision.models.resnet18,
        "RES34": torchvision.models.resnet34,
        "RES50": torchvision.models.resnet50,
    }
    if hyper_para.MODEL not in _CASE.keys():
        raise ValueError("Only support RES18, RES34, RES50")
    _kernel = _CASE.get(hyper_para.MODEL)
    return _kernel(num_classes=hyper_para.CLASS_CNT)


def make_dataset():
    _CASE = {
        "best": FullSpectroAudioSet,
        "auto-encoder": AutoEncodedAudioSet,
        "sound-power": SoundPowerAudioSet
    }
    if hyper_para.DATA_SET not in _CASE.keys():
        raise ValueError("Only support best, auto-encoder, sound-power")
    _kernel = _CASE.get(hyper_para.DATA_SET)
    return _kernel()


def make_dataloader(dataset):
    train, validate, test = tch_data.random_split(dataset, hyper_para.TRAIN_TEST_VALIDATE_SPLIT)
    return (
        tch_data.DataLoader(train, batch_size=hyper_para.BATCH_SIZE, shuffle=True),
        tch_data.DataLoader(validate, batch_size=hyper_para.BATCH_SIZE, shuffle=False),
        tch_data.DataLoader(test, batch_size=hyper_para.BATCH_SIZE, shuffle=False)
    )


def make_loss_function():
    return torch.nn.BCEWithLogitsLoss()


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=hyper_para.LEARNING_RATE)


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyper_para.STEP_SIZE, gamma=hyper_para.GAMMA)


def select_device():
    if hyper_para.DEVICE == "cuda" and "mac" in platform.platform().lower():
        return torch.device("mps")
    return torch.device(hyper_para.DEVICE)


def build_model_from_hyper_parameters(data_shape, hyper: Dict[str, Any]) -> Tuple[AudioEncoder, AudioDecoder]:
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


def build_model(data_shape) -> Tuple[AudioEncoder, AudioDecoder]:
    """
    Build the model
    Usage:
    >>> audio_encoder, audio_decoder = build_model(data_shape)
    Important: hyper_parameters_config.py must be imported before this function is called.
    :param data_shape: Shape of Encoder input.
    :return: Tuple[AudioEncoder, AudioDecoder]
    """
    _encoder, _decoder = build_model_from_hyper_parameters(data_shape, hyper_para.AUTO_ENCODER_MODEL)
    return _encoder, _decoder
