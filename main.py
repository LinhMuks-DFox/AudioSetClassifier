import os.path
import sys

import torch
import logging
import train_config
import train_prepare

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(train_config.DUMP_PATH, 'train.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.info

def train():
    pass

def one_step_loss():
    pass

def validate():
    pass

def test():
    pass


def main():
    pass