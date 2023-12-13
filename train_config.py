import datetime
import os.path
import platform

IN_DOCKER: bool = os.environ.get("IN_DOCKER_CONTAINER", False)

if "win" in (plf := platform.platform().lower()):
    TRAIN_DATA_SET_PATH: str = r"F:\DataSets\Audioset\balanced\segments"
    EVAL_DATE_SET_PATH: str = r"F:\DataSets\Audioset\eval\segments"

    EVAL_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_eval.json"
    TRAIN_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_train.json"

elif "mac" in plf:
    TRAIN_DATA_SET_PATH: str = r"/Volumes/PortMux/DataSet/Audioset/segments"
    EVAL_DATE_SET_PATH: str = r"/Volumes/PortMux/DataSet/AudiosetEval/segments"

    EVAL_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_eval.json"
    TRAIN_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_train.json"
else:
    TRAIN_DATA_SET_PATH: str = r"data/audio_set"
    EVAL_DATE_SET_PATH: str = r"data/audio_set_eval"

    EVAL_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_eval.json"
    TRAIN_DATA_SET_JSON: str = r"./subset_json/music_speech/sub_train.json"

PLATFORM: str = plf
DRY_RUN: bool = True
CPU_N_WORKERS: int = 23
TRAIN_ID = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
DUMP_PATH = f"./pth_bin/latent600-ablation/{TRAIN_ID}"
CLASS_LABELS_INDICES = r"subset_json/music_speech/sub_set_cls_label_idx.json"
AUTO_ENCODER_MODEL_PATH = r"pre_trained_encoder/2023-7-27-ablation/normal/encoder.pth"
TRAIN_CONFIG_SUMMARY = f"""
Train config summary of {TRAIN_ID}:
IN_DOCKER : {IN_DOCKER}
DATA_SET_PATH : {TRAIN_DATA_SET_PATH}
DRY_RUN : {DRY_RUN}
DUMP_PATH : {DUMP_PATH}
PLATFORM : {PLATFORM}
"""
