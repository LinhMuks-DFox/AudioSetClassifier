import torch
import torchvision
import hyper_para


def make_model():
    if hyper_para.MODEL not in hyper_para.MODEL_SET:
        raise ValueError("Model not valid")
    _kernel = {
        "RES18": torchvision.models.resnet18,
        "RES34": torchvision.models.resnet34,
        "RES50": torchvision.models.resnet50,
    }.get(hyper_para.MODEL)

    return _kernel(num_classes=hyper_para.CLASS_CNT)
