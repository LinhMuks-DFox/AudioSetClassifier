import platform
from typing import Tuple

import torch.nn
import torch.utils.data as tch_data
import torchvision

import hyper_para
import train_config
from lib.AutoEncoder.AudioDecoder import AudioDecoder
from lib.AutoEncoder.AudioEncoder import AudioEncoder
from lib.AutoEncoder.AutoEncoderPrepare import make_auto_encoder_from_hyperparameter
from src.AutoEncodedAudioSet import AutoEncodedAudioSet
from src.FullSpectroAudioSet import FullSpectroAudioSet
from src.SoundPowerAudioSet import SoundPowerAudioSet


def make_classifier():
    _CASE = {
        "RES18": torchvision.models.resnet18,
        "RES34": torchvision.models.resnet34,
        "RES50": torchvision.models.resnet50,
    }
    if hyper_para.MODEL not in _CASE.keys():
        raise ValueError("Only support RES18, RES34, RES50")
    _res_net = _CASE.get(hyper_para.MODEL)
    _projection = torch.nn.Conv2d(kernel_size=(1, 1), in_channels=1, out_channels=3)
    return torch.nn.Sequential(_projection, _res_net(num_classes=hyper_para.CLASS_CNT))


def make_dataset():
    if hyper_para.DATA_SET == "ideal":
        return FullSpectroAudioSet(
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
        )
    elif hyper_para.DATA_SET == "sound_power":
        return SoundPowerAudioSet(
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE
        )
    elif hyper_para.DATA_SET == "encoded":
        return AutoEncodedAudioSet(
            auto_encoder_hypers=hyper_para.AUTO_ENCODER_MODEL,
            encoder_model_path=train_config.AUTO_ENCODER_MODEL_PATH,
            path=train_config.DATA_SET_PATH,
            sound_track=hyper_para.AUDIO_PRE_TRANSFORM.get("sound_track"),
            orig_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("orig_freq"),
            new_freq=hyper_para.AUDIO_PRE_TRANSFORM.get("resample").get("new_freq"),
            n_fft=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("n_fft"),
            hop_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("hop_length"),
            win_length=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("win_length"),
            normalized=hyper_para.AUDIO_PRE_TRANSFORM.get("fft").get("normalized"),
            output_size=hyper_para.ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE
        )
    else:
        raise ValueError("Unknown data set type")


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
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=hyper_para.SCHEDULAR_GAMMA,
                                           gamma=hyper_para.SCHEDULAR_GAMMA)


def select_device():
    if hyper_para.DEVICE == "cuda" and "mac" in platform.platform().lower() and "arm" in platform.platform().lower():
        return torch.device("mps")
    return torch.device(hyper_para.DEVICE)


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
