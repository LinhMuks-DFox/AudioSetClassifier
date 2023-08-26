import torch.nn
import torch.utils.data as tch_data
import torchvision

import hyper_para
from src.AutoEncodedAudioSet import AutoEncodedAudioSet
from src.FullSpectroAudioSet import FullSpectroAudioSet
from src.SoundPowerAudioSet import SoundPowerAudioSet


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
