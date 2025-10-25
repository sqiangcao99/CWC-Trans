# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from tqdm import tqdm

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def do_perframe_det_batch_inference(cfg, args, model, device, logger):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE * 16,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    pred_scores = {}
    gt_targets = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-5]
            (det_score, history_score), consume = model.batch_inference(*[x.to(device) for x in data[:-5]])
            if cfg.CASCADE.ENABLE:
                det_score = det_score[-1]
            
            score = det_score.softmax(dim=-1).cpu().numpy()
            for bs, (session, query_indices, long_indices, num_frames) in enumerate(zip(*data[-4:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] == 0:
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

    if cfg.DDP.ENABLE:
        torch.distributed.barrier(device_ids=[args.local_rank])
        pred_dict = [None for _ in range(args.world_size)] 
        gt_dict = [None for _ in range(args.world_size)] 
        torch.distributed.all_gather_object(pred_dict, pred_scores)
        torch.distributed.all_gather_object(gt_dict, gt_targets)
    
    pkl.dump({
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
    }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))
