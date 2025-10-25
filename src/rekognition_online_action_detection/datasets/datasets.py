
__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch
import torch.distributed as dist
import torch.utils.data as data
from typing import Any, Optional

from rekognition_online_action_detection.utils.helper import torch_distributed_zero_first
from rekognition_online_action_detection.utils.registry import Registry
from rekognition_online_action_detection.utils.sampler import (
    BalancedBatchSampler,
    DistBalancedBatchSampler,
    OrderedDistributedSampler,
)


DATA_LAYERS = Registry()

def build_dataset(cfg, phase, tag=''):
    data_layer = DATA_LAYERS[cfg.MODEL.MODEL_NAME + tag + cfg.DATA.DATA_NAME]
    
    return data_layer(cfg, phase)

def build_data_loader(cfg, args, phase):
    sampler = None

    if cfg.DDP.ENABLE: 
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        with torch_distributed_zero_first(global_rank):
            dataset = build_dataset(cfg, phase)

        if phase == 'train': 
            torch.distributed.barrier(device_ids=[args.local_rank])
            ds_m = [None for _ in range(args.world_size)]
            dist.all_gather_object(ds_m, dataset)
            dataset = ds_m[0]

    else:
        dataset = build_dataset(cfg, phase)

    if cfg.DDP.ENABLE:
        if phase == 'train':
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler = OrderedDistributedSampler(dataset)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.BATCH_SIZE // args.world_size,
        shuffle=True if phase == 'train' and sampler == None else False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        sampler=sampler, 
    )
    
    return data_loader