import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
    """
    Với đào tạo trên 1 GPU, không cần thiết lập distributed.
    Trả về rank=0 và world_size=1.
    """
    rank = 0
    world_size = 1
    return rank, world_size
