import platform
from typing import Tuple

import torch.nn
import torch.utils.data as tch_data
import torchvision
import os
import hyper_para
import train_config
from lib.AutoEncoder.AudioDecoder import AudioDecoder
from lib.AutoEncoder.AudioEncoder import AudioEncoder
from lib.AutoEncoder.AutoEncoderPrepare import make_auto_encoder_from_hyperparameter
from src.AutoEncodedAudioSet import AutoEncodedAudioSet
from src.FullSpectroAudioSet import FullSpectroAudioSet
from src.SoundPowerAudioSet import SoundPowerAudioSet


def compose_path(dump_path: str = None, dataset_type: str = None, file_name=None) -> str:
    dump_path = dump_path if dump_path is not None else train_config.DUMP_PATH
    dataset_type = dataset_type if dataset_type is not None else hyper_para.DATA_SET
    if file_name is None:
        return os.path.join(dump_path, dataset_type)
    return os.path.join(dump_path, dataset_type, file_name)


def make_classifier(model_type: str = None, class_cnt: int = None):
    model_type = model_type if model_type is not None else hyper_para.MODEL
    class_cnt = class_cnt if class_cnt is not None else hyper_para.CLASS_CNT
    _CASE = {
        "RES18": torchvision.models.resnet18,
        "RES34": torchvision.models.resnet34,
        "RES50": torchvision.models.resnet50,
    }
    if hyper_para.MODEL not in _CASE.keys():
        raise ValueError("Only support RES18, RES34, RES50")
    _res_net = _CASE.get(model_type)
    _projection = torch.nn.Conv2d(kernel_size=(1, 1), in_channels=1, out_channels=3)
    return torch.nn.Sequential(_projection, _res_net(num_classes=class_cnt))


def make_dataset(path: str = None, dataset_type: str = None):
    path = path if path is not None else train_config.TRAIN_DATA_SET_PATH
    dataset_type = dataset_type if dataset_type is not None else hyper_para.DATA_SET
    if dataset_type == "ideal":
        return FullSpectroAudioSet(
            path=path,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
        )
    elif dataset_type == "sound_power":
        return SoundPowerAudioSet(
            path=path,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE
        )
    elif dataset_type == "encoded":
        return AutoEncodedAudioSet(
            auto_encoder_hypers=hyper_para.AUTO_ENCODER_MODEL,
            encoder_model_path=train_config.AUTO_ENCODER_MODEL_PATH,
            path=path,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE,
            transform_device=select_device(hyper_para.DATA_TRANSFORM_DEVICE),
        )
    else:
        raise ValueError("Unknown data set type")


def make_dataloader(dataset, batch_size: int = None):
    train, validate, test = tch_data.random_split(dataset, hyper_para.TRAIN_TEST_VALIDATE_SPLIT)
    batch_size = batch_size if batch_size is not None else hyper_para.BATCH_SIZE
    return (
        tch_data.DataLoader(train, batch_size=batch_size, shuffle=True),
        tch_data.DataLoader(validate, batch_size=batch_size, shuffle=False),
        tch_data.DataLoader(test, batch_size=batch_size, shuffle=False)
    )


def make_loss_function(loss_type: str = None, loss_args: dict = None):
    loss_type = loss_type if loss_type is not None else hyper_para.LOSS_FUNCTION.get("name")
    loss_args = loss_args if loss_args is not None else hyper_para.LOSS_FUNCTION.get("arg")
    return {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
        "BCELoss": torch.nn.BCELoss,
        "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
    }.get(loss_type)(**loss_args)


def make_optimizer(model, optimizer_type: str = None, lr: float = None):
    optimizer_type = optimizer_type if optimizer_type is not None else hyper_para.OPTIMIZER
    lr = lr if lr is not None else hyper_para.LEARNING_RATE
    return {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }.get(optimizer_type)(model.parameters(), lr=lr)


def make_scheduler(optimizer):
    return {
        "StepLR": torch.optim.lr_scheduler.StepLR
    }.get(hyper_para.SCHEDULER)(optimizer, step_size=hyper_para.SCHEDULAR_STEP_SIZE, gamma=hyper_para.SCHEDULAR_GAMMA)


def select_device(device=None):
    _device = device if device is not None else hyper_para.TRAIN_DEVICE
    if "cuda" in _device and "mac" in platform.platform().lower():
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device(_device)


def make_auto_encoder_model(data_shape) -> Tuple[AudioEncoder, AudioDecoder]:
    """
    Build the model
    Usage:
    >>> audio_encoder, audio_decoder = make_auto_encoder_model(data_shape)
    Important: hyper_parameters_config.py must be imported before this function is called.
    :param data_shape: Shape of Encoder input.
    :return: Tuple[AudioEncoder, AudioDecoder]
    """
    _encoder, _decoder = make_auto_encoder_from_hyperparameter(data_shape, hyper_para.AUTO_ENCODER_MODEL)
    return _encoder, _decoder


def set_torch_random_seed():
    _device = select_device()
    torch.manual_seed(hyper_para.RANDOM_SEED)
    if "cuda" in f"{_device}":
        torch.cuda.manual_seed(hyper_para.RANDOM_SEED)
        torch.cuda.manual_seed_all(hyper_para.RANDOM_SEED)
    if "mps" in f"{_device}":
        torch.mps.manual_seed(hyper_para.RANDOM_SEED)
