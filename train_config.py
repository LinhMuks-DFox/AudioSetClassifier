import datetime
import os.path
import platform

IN_DOCKER: bool = os.environ.get("IN_DOCKER_CONTAINER", False)

if "win" in (plf := platform.platform().lower()):
    DATA_SET_PATH: str = r"F:\DataSets\Audioset\balanced\segments\AudioSet.json"
elif "mac" in plf:
    DATA_SET_PATH: str = r"/Volumes/PortMux/DataSet/Audioset/segments/AudioSet.json"
else:
    DATA_SET_PATH: str = r"data/audio_set/AudioSet.json"

PLATFORM: str = plf
DRY_RUN: bool = True
CPU_N_WORKERS: int = 23
TRAIN_ID = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
DUMP_PATH = f"./pth_bin/{TRAIN_ID}"
AUTO_ENCODER_MODEL_PATH = r"pre_trained_encoder/2023-08-04-09-49-03/encoder.pth"
TRAIN_CONFIG_SUMMARY = f"""
Train config summary of {TRAIN_ID}:
IN_DOCKER : {IN_DOCKER}
DATA_SET_PATH : {DATA_SET_PATH}
DRY_RUN : {DRY_RUN}
DUMP_PATH : {DUMP_PATH}
PLATFORM : {PLATFORM}
"""
