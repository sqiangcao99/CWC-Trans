# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    'setup_random_seed',
    'setup_environment',
]

import os
import random
import logging

import torch
import numpy as np

from torch.nn.parallel import DistributedDataParallel as NativeDDP


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_environment(cfg, args):
    if "WORLD_SIZE" in os.environ.keys():
        cfg.DDP.ENABLE=int(os.environ['WORLD_SIZE']) > 1
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    
    if cfg.DDP.ENABLE:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = 'cuda:%d' % args.local_rank
        print('Local Rank:{}'.format(args.local_rank), 'Device', args.device)
        
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', )
        torch.distributed.barrier(device_ids=[args.local_rank])
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')

    assert args.rank >= 0

    if cfg.SEED is not None:
        setup_random_seed(cfg.SEED)
    return args.device