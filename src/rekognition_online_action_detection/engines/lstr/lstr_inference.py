# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import einops
import os.path as osp
from bisect import bisect_right

import torch
import numpy as np

from rekognition_online_action_detection.evaluation import compute_result

from ..base_inferences.perframe_det_batch_inference import do_perframe_det_batch_inference
from ..engines import INFERENCES as registry

@registry.register('LSTR')
def do_lstr_batch_inference(cfg,
                            args, 
                            model,
                            device,
                            logger):
    if cfg.MODEL.LSTR.INFERENCE_MODE == 'stream':
        do_lstr_stream_inference(cfg,
                                 model,
                                 device,
                                 logger)
    else:
        do_perframe_det_batch_inference(cfg,
                                        args,
                                        model,
                                        device,
                                        logger)
