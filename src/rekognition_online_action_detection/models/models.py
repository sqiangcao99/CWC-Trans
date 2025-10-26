# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_model']

import torch
from rekognition_online_action_detection.utils.registry import Registry
from torch.nn.parallel import DistributedDataParallel as NativeDDP

META_ARCHITECTURES = Registry()


def build_model(cfg, args, device=None):
    model = META_ARCHITECTURES[cfg.MODEL.MODEL_NAME](cfg)
    from .weights_init import weights_init
    model.apply(weights_init)

    model.cuda(args.local_rank)
    
    if cfg.DDP.ENABLE and cfg.DDP.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)

        if args.local_rank == 0:
            print('Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if cfg.DDP.ENABLE:
        model = NativeDDP(model, device_ids=[args.local_rank],)
    
    return model