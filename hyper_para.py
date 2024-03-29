import numpy as np
import torch

import train_config

MODEL = "RES18"
CLASS_CNT = 2
TRAIN_DEVICE = "cuda:0"
DATA_TRANSFORM_DEVICE = "cuda:0"
BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE = 1e-6
SCHEDULER = "MultiStepLR"
SCHEDULER_PARAMETER = {
    "gamma": 0.9,
    # "milestones": [50, 100, 125, 145, 160, 170, 180, 175, 180, 185, 190, 195, 197],
    "milestones": [10, 20, 30, 40, 45, 50, 55],
    # "milestones": [20, 30, 45,],

}
DATA_SET = "sound_power"  # "ideal", "sound_power", "encoded"
TRAIN_TEST_VALIDATE_SPLIT = [0.8, 0.1, 0.1]
VALIDATE_TEST_SPLIT = [0.5, 0.5]
OPTIMIZER = "Adam"
CHECK_POINT_INTERVAL = 10
ONT_HOT_LABEL = True
MODEL_SELECT_MILESTONE = 20  # from milestone 10, selected

LOSS_FUNCTION = {
    "name": "CrossEntropy",
    "arg": {"reduction": "mean"}
}
AUDIO_PRE_TRANSFORM = {
    "sound_track": "mix",
    "resample": {
        "orig_freq": 44100,
        "new_freq": 16000
    },
    "fft": {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 128,
        "normalized": True,
    }

}
CAMERA_RESPONSE = {
    "source_sample_rate": 15,
    "camera_sample_rate": 30,
    "temperature": 0.1
}
LIGHT_PROPA = {
    "distance": 1,  # 5m
    "bias": 0,  # 10% bias from env
    "std": 0
}
AUTO_ENCODER_MODEL = {
    "conv_kernel_size": np.array([[3, 3],
                                  [5, 5],
                                  [11, 11],
                                  [7, 7],
                                  [3, 3],
                                  [5, 5]]),
    "conv_padding": np.array(((0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0),)),
    "conv_stride": np.array(((1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1),)),
    "conv_dilation": np.array(((1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),)),
    "encoder_output_feature": 300,  # ((CAMERA_FPS * FIX_LENGTH) // 3) * 2 * BLINKY_LED_CNT
    "convolution_times": 6,
    "conv_output_channel": np.array([1, 8, 64, 32, 8, 1]),
    "conv_type": torch.nn.Conv2d
}
ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE = (1, 4, 300)  # 5s -> 300, 10s -> 600; 10 * 80 => 80floats per second
# region DRY_RUN
DRY_RUN_EPOCHS = 10
DYR_RUN_BATCH_SIZE = 30
DRY_RUN_DATE_SET_LENGTH: int = 80
DRY_MODEL_SELECT_MILESTONE = 2
TRAIN_HYPER_PARA_SUMMARY = \
    f"""Hyperparameter summary: 
model: {MODEL}
class count: {CLASS_CNT}
train device(in-hyperparameter): {TRAIN_DEVICE}
data transforming device(in-hyperparameter): {DATA_TRANSFORM_DEVICE}
batch size: {BATCH_SIZE if not train_config.DRY_RUN else DYR_RUN_BATCH_SIZE}
optimizer: {OPTIMIZER}
loss function: {LOSS_FUNCTION}
schedular: {SCHEDULER}
epochs: {EPOCHS if not train_config.DRY_RUN else DRY_RUN_EPOCHS}
learning rate: {LEARNING_RATE}
schedular parameter: {SCHEDULER_PARAMETER}
data set: {DATA_SET}
train test validate split: {TRAIN_TEST_VALIDATE_SPLIT}
validate test split: {VALIDATE_TEST_SPLIT}
model select milestone: {MODEL_SELECT_MILESTONE if not train_config.DRY_RUN else DRY_MODEL_SELECT_MILESTONE}
""" + f"dry run dataset length: {DRY_RUN_DATE_SET_LENGTH}" if train_config.DRY_RUN else ""

RANDOM_SEED = 65536
