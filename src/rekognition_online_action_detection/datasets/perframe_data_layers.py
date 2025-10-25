# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data

from .datasets import DATA_LAYERS as registry


@registry.register('LSTRTHUMOS')
@registry.register('LSTRTVSeries')
class LSTRDataLayer(data.Dataset):
    """Data layer for LSTR model training on THUMOS and TVSeries datasets."""

    def __init__(self, cfg: Any, phase: str = 'train'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = int(cfg.MODEL.LSTR.WORK_MEMORY_LENGTH)

        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.training = phase == 'train'

        self.move_step = cfg.DATA.MOVE_STEP
        if not self.training:
            self.move_step = 32

        self.aug = cfg.DATA.AUG.ENABLE
        self.aug_type = cfg.DATA.AUG.TYPE

        if cfg.DATA.AUG.ENABLE:
            if cfg.DATA.AUG.TYPE == 'random drop':
                self.rng_rate = cfg.DATA.AUG.RANDOM_DROP_RATE
            elif cfg.DATA.AUG.TYPE == 'first drop':
                self.first_rate = cfg.DATA.AUG.FIRST_DROP_RATE 
                self.first_len = cfg.DATA.AUG.FIRST_DROP_LENGTH

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            seed = np.random.randint(self.work_memory_length) if self.training else 0
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.move_step), 
                range(seed + self.work_memory_length, target.shape[0], self.move_step)):
                
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end],
                ]) 

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index: int) -> Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
            Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
        ]:
        session, work_start, work_end, target = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')
        
        targets_all = np.load(
            osp.join(self.data_root, self.target_perframe, session + '.npy'), mmap_mode='r')
        target_current = target[::self.work_memory_sample_rate]
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]
            target_long = targets_all[long_indices]
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
            
            if self.aug and self.training:
                if self.aug_type == 'random drop':
                    drop_mask = np.ones_like(memory_key_padding_mask)
                    drop_mask = torch.from_numpy(drop_mask)
                    drop_mask = F.dropout(drop_mask, p=self.rng_rate)
                    drop_mask = drop_mask.numpy()
                    drop_mask = (drop_mask <0.5)
                    memory_key_padding_mask[drop_mask] = float('-inf')
                elif self.aug_type == 'first drop':
                    stop_index = np.random.randint(self.first_len)
                    if np.random.rand() <= self.first_rate:
                        memory_key_padding_mask[:stop_index] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target_current, target_long
        else:
            return fusion_visual_inputs, fusion_motion_inputs, target_current, target_long
        
    def __len__(self):
        return len(self.inputs)

@registry.register('LSTRBatchInferenceTHUMOS')
@registry.register('LSTRBatchInferenceTVSeries')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = int(cfg.MODEL.LSTR.WORK_MEMORY_LENGTH)
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        self.shift_bound = int(cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE - self.work_memory_num_samples / self.long_memory_sample_rate) 
        self.move_step = cfg.DATA.MOVE_STEP
        self.window_size = int(cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE)

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            for work_start, work_end in zip(
                range(0, target.shape[0] + 1),
                range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end], target.shape[0]
                ]) 

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index] 
        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        target = target[::self.work_memory_sample_rate]
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        shift_steps = ((work_start // self.long_memory_sample_rate)) % self.window_size
        if shift_steps > 0:
            remained_steps = self.window_size - shift_steps
            long_visual_inputs = np.concatenate((long_visual_inputs[:remained_steps], long_visual_inputs[-shift_steps:], long_visual_inputs[remained_steps:-shift_steps]), axis=0)
            long_motion_inputs = np.concatenate((long_motion_inputs[:remained_steps],  long_motion_inputs[-shift_steps:], long_motion_inputs[remained_steps:-shift_steps]), axis=0)    
            memory_key_padding_mask = np.concatenate((memory_key_padding_mask[:remained_steps], memory_key_padding_mask[-shift_steps:], memory_key_padding_mask[remained_steps:-shift_steps]), axis=0)
        
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)