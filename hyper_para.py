import torch
import numpy as np

import train_config

MODEL = "RES34"
CLASS_CNT = 527  # Audio set contains 527 class labels
DEVICE = "cuda:0"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5
SCHEDULAR_GAMMA = 0.99
DATA_SET = "ideal"  # "ideal", "sound_power", "encoded"
TRAIN_TEST_VALIDATE_SPLIT = [0.8, 0.1, 0.1]
OPTIMIZER = "Adam"
LOSS_FUNCTION = "BCEWithLogitsLoss"
SCHEDULER = "StepLR"
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

AUTO_ENCODER_MODEL = {
    "conv_kernel_size": np.array([[3, 3],
                                  [5, 5],
                                  [3, 3],
                                  [5, 5],
                                  [3, 3],
                                  [3, 3]]),

    "conv_padding": np.array(((0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0),
                              (0, 0))),
    "conv_stride": np.array(((1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1),
                             (1, 1))),
    "conv_dilation": np.array(((1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1))),
    "encoder_output_feature": 400,  # ((CAMERA_FPS * FIX_LENGTH) // 3) * 2 * BLINKY_LED_CNT
    "convolution_times": 6,
    "conv_output_channel": np.array([1, 8, 32, 16, 8, 1]),
    "conv_type": torch.nn.Conv2d
}
ENCODED_AND_SOUND_POWER_DATASET_RESHAPE_SIZE = (10, 80)  # 5s -> 400, 10s -> 800; 10 * 80 => 80floats per second
# region DRY_RUN
DRY_RUN_EPOCHS = 3
DYR_RUN_BATCH_SIZE = 30
DRY_RUN_DATE_SET_LENGTH: int = 80

TRAIN_HYPER_PARA_SUMMARY = \
    f"""Hyperparameter summary: 
model: {MODEL}
class count: {CLASS_CNT}
device(in-hyperparameter): {DEVICE}
batch size: {BATCH_SIZE}
optimizer: {OPTIMIZER}
loss function: {LOSS_FUNCTION}
schedular: {SCHEDULER}
epochs: {EPOCHS}
learning rate: {LEARNING_RATE}
schedular gamma: {SCHEDULAR_GAMMA}
data set: {DATA_SET}
train test validate split: {TRAIN_TEST_VALIDATE_SPLIT}
"""
DRY_RUN_MESSAGE = \
    f"""dry run epochs: {DRY_RUN_EPOCHS}
dry run batch size: {DYR_RUN_BATCH_SIZE}
dry run data set length: {DRY_RUN_DATE_SET_LENGTH}
"""
if train_config.DRY_RUN:
    TRAIN_HYPER_PARA_SUMMARY += DRY_RUN_MESSAGE

RANDOM_SEED = 0
