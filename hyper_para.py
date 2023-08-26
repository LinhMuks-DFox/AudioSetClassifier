import torch

MODEL = "RES18"
CLASS_CNT = 527  # Audio set contains 527 class labels
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5
GAMMA = 0.99
DATA_SET = "best"
TRAIN_TEST_VALIDATE_SPLIT = [0.8, 0.1, 0.1]
