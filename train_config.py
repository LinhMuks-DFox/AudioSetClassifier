import os.path
import sys
import datetime

IN_DOCKER: bool = os.environ.get("IN_DOCKER_CONTAINER", False)
if "win" in sys.platform or not IN_DOCKER:
    DATA_SET_PATH: str = r"F:\DataSets\Audioset\balanced\segments\AudioSet.json"
else:
    DATA_SET_PATH: str = r"data/audio_set/AudioSet.json"
DRY_RUN: bool = False
DRY_RUN_DATE_SET_LENGTH: int = 80

DUMP_PATH = f"pth_bin/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
if not os.path.exists(DUMP_PATH):
    os.mkdir(DUMP_PATH)
else:
    raise RuntimeError("DUMP PATH ALREADY EXISTS")