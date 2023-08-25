import torch.cuda

MODEL = "RES18"
MODEL_SET = {
    "RES18",
    "RES50"
}
CLASS_CNT = 527  # Audio set contains 527 class labels
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5