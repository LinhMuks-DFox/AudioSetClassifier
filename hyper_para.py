import torch
import numpy as np

MODEL = "RES18"
CLASS_CNT = 527  # Audio set contains 527 class labels
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5
GAMMA = 0.99
DATA_SET = "best"
TRAIN_TEST_VALIDATE_SPLIT = [0.8, 0.1, 0.1]

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
