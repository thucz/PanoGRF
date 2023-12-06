import argparse

from train.trainer_mv import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/train/gen/neuray_gen_depth_train.yaml')
parser.add_argument('--data_idx', type=int, default=0, help='data_idx of test data')


flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg), flags.data_idx)
trainer.run()