import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

# Định nghĩa các tham số cần thiết
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)

def main():
    args = parser.parse_args()
    # Load cấu hình từ file YAML (bao gồm các thông số như learning rate, batch_size, crop_size, ...)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # Khởi tạo logger và SummaryWriter cho TensorBoard
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    rank = 0
    world_size = 1

    # In ra cấu hình tổng hợp
    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

if __name__ == '__main__':
    main()