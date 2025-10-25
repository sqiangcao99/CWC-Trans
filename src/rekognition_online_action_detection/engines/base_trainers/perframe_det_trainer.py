# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import einops
import torch.distributed as dist
from tqdm import tqdm

import torch
import torch.nn as nn

from rekognition_online_action_detection.evaluation import compute_result
from rekognition_online_action_detection.utils.helper import reduce_tensor
from rekognition_online_action_detection.datasets import build_data_loader

def do_perframe_det_train(cfg,
                          args, 
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          device,
                          checkpointer,
                          logger):
    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        current_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        as_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        cascade_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        oas_cas_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        
        current_pred_scores = []
        current_gt_targets = []
        start = time.time()
        consumes = 0

        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            data_loader = data_loaders[phase]
            
            with torch.set_grad_enabled(training):
                if args.local_rank == 0:
                    pbar = tqdm(range(1, 1 + len(data_loader)),
                        desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                
                for batch_idx, data in enumerate(data_loader, start=1): 
                    batch_size = data[0].shape[0]
                    det_target = data[-2].to(args.device)
                    long_target = data[-1].to(args.device)
                    history_mask = data[-3].to(args.device)    
                    (det_score, history_score), extend_index, consume = model(*[x.to(device) for x in data[:-2]])
                    
                    consumes += consume
                    extend_target = det_target[:,extend_index,:]
                    
                    long_movement = long_target[:,:32,:]
                    fixed_movement = long_target[:,32:,:]
                    long_movement = torch.cat((long_movement, extend_target), dim = 1)
                    long_movement = long_movement.unfold(dimension=1, size=32, step=1).permute(0,1,3,2)[:,:-1]

                    fixed_movement = einops.repeat(fixed_movement, 'bsz nl p-> bsz nh nl p', nh=8)
                    long_target = torch.cat((long_movement, fixed_movement), dim=2) 
                    
                    mask_movement = history_mask[:,:32]
                    fixed_mask = history_mask[:,32:]

                    mask_extend = torch.zeros(history_mask.shape[0], 8).to(history_mask.device)
                    mask_movement = torch.cat((mask_movement, mask_extend), dim=1)
                    mask_movement = mask_movement.unfold(dimension=1, size=32, step=1)[:,:-1]
                    fixed_mask = einops.repeat(fixed_mask, 'bsz nl-> bsz nh nl', nh=8)
                    history_mask = torch.cat((mask_movement, fixed_mask),dim=2)
                
                    long_target = einops.rearrange(long_target, 'b nh l p-> (b nh) l p')
                    history_mask = einops.rearrange(history_mask, 'b nh nl->(b nh) nl')
                    history_mask = einops.rearrange(history_mask, 'b nl-> (b nl)')

                    loss = torch.tensor(0.).to(args.device)
                    as_loss = torch.tensor(0.).to(args.device)
                    cascade_loss = torch.tensor(0.).to(args.device)
                    oas_cas_loss = torch.tensor(0.).to(args.device)

                    if cfg.OAS.ENABLE and history_score != None:
                        if cfg.CASCADE.ENABLE and cfg.CASCADE.OAS_ENBALE:
                            oas_cas_score = history_score[1:]
                            history_score = history_score[0]

                            oas_cas_target = einops.repeat(long_target, 'bsz nl p -> (ns bsz) nl p', ns = cfg.CASCADE.SA.NUM_STAGES-1)
                            oas_cas_mask = einops.repeat(history_mask, 'bnl -> (ns bnl)', ns = cfg.CASCADE.SA.NUM_STAGES-1)
                            
                            oas_cas_target = oas_cas_target.reshape(-1, cfg.DATA.NUM_CLASSES)[history_mask == 0]                    
                            oas_cas_score = oas_cas_score.reshape(-1, cfg.DATA.NUM_CLASSES)[history_mask == 0]    

                            oas_cas_loss  = criterion['MCE'](oas_cas_score, oas_cas_target)
                            loss = loss + 0.4 * oas_cas_loss
                          
                        long_target = long_target.reshape(-1, cfg.DATA.NUM_CLASSES)[history_mask == 0]
                        history_score = history_score.reshape(-1, cfg.DATA.NUM_CLASSES)[history_mask == 0]
                        as_loss = criterion['MCE'](history_score, long_target)
                        loss = loss + 0.2 * as_loss

                    if cfg.CASCADE.ENABLE:
                        if phase == 'train':
                            current_score = det_score[0]
                            cascade_score = det_score[1:]
                        else:
                            current_score = det_score[-1]
                            cascade_score = det_score[1:]
                        
                        cascade_target = einops.repeat(det_target, 'bsz nl p -> (ns bsz) nl p', ns = cfg.CASCADE.SA.NUM_STAGES-1)
                        cascade_target = cascade_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        current_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)

                        cascade_score = einops.rearrange(cascade_score, 'ns bsz nl p -> (ns bsz) nl p', ns = cfg.CASCADE.SA.NUM_STAGES-1)
                        current_score = current_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        cascade_score = cascade_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        
                        cascade_loss = criterion['MCE'](cascade_score, cascade_target)
                        current_loss = criterion['MCE'](current_score, current_target)
                        loss = loss + 1.0 * current_loss + 0.7 * cascade_loss
                    else:
                        current_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        current_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)

                        current_loss = criterion['MCE'](current_score, current_target)
                        loss = loss + current_loss

                    if training:           
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:
                        if cfg.DDP.ENABLE:
                            current_score_m = [torch.empty_like(current_score) for _ in range(args.world_size)]
                            torch.distributed.barrier(device_ids=[args.local_rank])
                            
                            torch.distributed.all_gather(current_score_m, current_score)
                            current_score_m = torch.cat(current_score_m, dim=0)
                            
                            current_target_m = [torch.empty_like(current_target) for _ in range(args.world_size)]
                            torch.distributed.barrier(device_ids=[args.local_rank])
                            torch.distributed.all_gather(current_target_m, current_target)
                            current_target_m = torch.cat(current_target_m, dim=0)
                        else:
                            current_score_m = current_score
                            current_target_m = current_target
                            
                        current_score_m = current_score_m.softmax(dim=1).cpu().tolist()
                        current_target_m = current_target_m.cpu().tolist()

                        current_pred_scores.extend(current_score_m)
                        current_gt_targets.extend(current_target_m)

                    if cfg.DDP.ENABLE: 
                        loss = reduce_tensor(loss, args.world_size)
                        current_loss = reduce_tensor(current_loss, args.world_size)
                        as_loss = reduce_tensor(as_loss, args.world_size)
                        cascade_loss = reduce_tensor(cascade_loss, args.world_size)
                        oas_cas_loss = reduce_tensor(oas_cas_loss, args.world_size)

                    if args.local_rank == 0:
                        pbar.set_postfix({
                            'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                            'current_loss': '{:.5f}'.format(current_loss.item()), 
                            'as_loss': '{:.5f}'.format(as_loss.item()), 
                            'cas_loss': '{:.5f}'.format(cascade_loss.item()), 
                            'oas_cas_loss': '{:.5f}'.format(oas_cas_loss.item()), 
                        })
                        pbar.update(1)
                    
                    current_losses[phase] += current_loss.item() * batch_size * args.world_size
                    as_losses[phase] += as_loss.item() * batch_size * args.world_size
                    cascade_losses[phase] += cascade_loss.item() * batch_size * args.world_size
                    oas_cas_losses[phase] += oas_cas_loss.item() * batch_size * args.world_size
                     
                    if cfg.DDP.ENABLE:
                        torch.distributed.barrier(device_ids=[args.local_rank])
                    
            if args.local_rank == 0:
                pbar.close()   
            
        end = time.time() 

        if args.local_rank == 0:
            log = []
            log.append('Epoch {:2}'.format(epoch))
            log.append('train current_loss: {:.5f} as_loss: {:.5f} cas_loss: {:.5f} oas_cas_loss: {:.5f}'.format(
                current_losses['train'] / len(data_loaders['train'].dataset),
                as_losses['train'] / len(data_loaders['train'].dataset),
                cascade_losses['train'] / len(data_loaders['train'].dataset),
                oas_cas_losses['train'] / len(data_loaders['train'].dataset),
            ))
            
            if 'test' in cfg.SOLVER.PHASES:
                det_result = compute_result['perframe'](
                    cfg,
                    current_gt_targets,
                    current_pred_scores,
                )

            log.append('test current_loss: {:.5f} as_loss: {:.5f} cas_loss: {:.5f} oas_cas_loss: {:.5f} det_mAP: {:.5f}'.format(
                current_losses['test'] / len(data_loaders['test'].dataset),
                as_losses['test'] / len(data_loaders['test'].dataset),
                cascade_losses['test'] / len(data_loaders['test'].dataset),
                oas_cas_losses['test'] / len(data_loaders['test'].dataset), 
                det_result['mean_AP'],
            ))

            log.append('running time: {:.2f} sec'.format(
                end - start,
            ))

            log.append('infer time: {:.2f} sec'.format(
                consumes,
            ))
            consumes = 0
            logger.info(' | '.join(log))
            
            checkpointer.save(epoch, model, optimizer)

        data_loaders['train'] = build_data_loader(cfg, args, 'train')
