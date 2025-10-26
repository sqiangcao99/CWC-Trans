# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from . import transformer as tr
from . import decoder as dc

import numpy as np


from .models import META_ARCHITECTURES as registry
from .feature_head import build_feature_head
from ..utils import gen_gass_pos_weights
from .self_compression import SelfCompression
from .history_decoder import HistoryDecoder
from .cascade import TCNCascade, SACascade

@registry.register('LSTR')
class LSTR(nn.Module):
    def __init__(self, cfg):
        super(LSTR, self).__init__()

        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg, is_long=True)

        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg, is_long=False)

        self.d_model = self.feature_head_work.d_model
        self.d_model_long = self.feature_head_long.d_model

        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dim_feedforward_long = cfg.MODEL.LSTR.DIM_FEEDFORWARD_LONG
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)
        self.pos_encoding_long = tr.PositionalEncoding(cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES_LONG, self.dropout) 

        self.current_extend = int(cfg.DATA.FPS * cfg.MODEL.LSTR.WORK_MEMORY_SECONDS)
        self.history_extend = int(self.long_memory_num_samples + np.around(self.current_extend / cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE)) - 1 
        self.history_length = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.history_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.num_history = int(self.current_extend / cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE)
        self.window_size = cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE
        self.num_window = int(self.history_length / self.window_size)
        self.history_encoder = SelfCompression(cfg)

        self.oas = cfg.OAS.ENABLE
        if self.oas:
            self.his_decoder = HistoryDecoder(
                                output_tokens = cfg.OAS.OUTPUT_TOKENS,
                                num_layers = cfg.OAS.NUM_LAYERS,
                                embed_dim = cfg.OAS.EMB_DIM, 
                                num_heads = cfg.OAS.NUM_HEADS,
                                window_size = cfg.OAS.WINDOW_SIZE,
                                num_patches = cfg.OAS.NUM_PATCHES,
                                num_tsm = cfg.OAS.NUM_TSM
            )
        self.cascade = cfg.CASCADE.ENABLE 
        self.cascade_shared = cfg.CASCADE.OAS_SHARED
        if self.cascade:
            self.oas_cascade = cfg.CASCADE.OAS_ENBALE
            if cfg.CASCADE.TYPE == 'tcn':
                self.cas_module = TCNCascade(
                                num_stages = cfg.CASCADE.NUM_STAGES,
                                num_layers = cfg.CASCADE.TCN.NUM_LAYERS,
                                num_f_maps = cfg.CASCADE.TCN.HIDDEN_DIM,
                                dim = cfg.CASCADE.TCN.IN_DIM,
                                num_classes = cfg.CASCADE.TCN.OUT_DIM,
                                dropout = cfg.CASCADE.TCN.DROPOUT
                )
            elif cfg.CASCADE.TYPE == 'SA':
                if cfg.CASCADE.OAS_SHARED:

                    self.cas_module = SACascade(
                                    num_stages = cfg.CASCADE.SA.NUM_STAGES,
                                    num_layers = cfg.CASCADE.SA.NUM_LAYERS,
                                    hidden_dim = cfg.CASCADE.SA.HIDDEN_DIM,
                                    num_classes = cfg.CASCADE.SA.NUM_CLASSES,
                                    num_heads = cfg.CASCADE.SA.NUM_HEADS,
                                    dropout = cfg.CASCADE.SA.DROPOUT,
                                    activation = cfg.CASCADE.SA.ACTIVATION,
                                    norm = cfg.CASCADE.SA.NORM,
                                    short_cut = cfg.CASCADE.SA.SHORTCUT,
                    )
                else: 
                    self.cas_module = SACascade(
                                    num_stages = cfg.CASCADE.SA.NUM_STAGES,
                                    num_layers = cfg.CASCADE.SA.NUM_LAYERS,
                                    hidden_dim = cfg.CASCADE.SA.HIDDEN_DIM,
                                    num_classes = cfg.CASCADE.SA.NUM_CLASSES,
                                    num_heads = cfg.CASCADE.SA.NUM_HEADS,
                                    dropout = cfg.CASCADE.SA.DROPOUT,
                                    activation = cfg.CASCADE.SA.ACTIVATION,
                                    norm = cfg.CASCADE.SA.NORM,
                                    short_cut = cfg.CASCADE.SA.SHORTCUT,)
                    self.cas_module_h = SACascade(
                                    num_stages = cfg.CASCADE.SA.NUM_STAGES,
                                    num_layers = cfg.CASCADE.SA.NUM_LAYERS,
                                    hidden_dim = cfg.CASCADE.SA.HIDDEN_DIM,
                                    num_classes = cfg.CASCADE.SA.NUM_CLASSES,
                                    num_heads = cfg.CASCADE.SA.NUM_HEADS,
                                    dropout = cfg.CASCADE.SA.DROPOUT,
                                    activation = cfg.CASCADE.SA.ACTIVATION,
                                    norm = cfg.CASCADE.SA.NORM,
                                    short_cut = cfg.CASCADE.SA.SHORTCUT,)

        if self.long_enabled:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, 16, self.dim_feedforward,
                self.dropout, self.activation, qdim = self.d_model, kdim = self.d_model)
            self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        self.classifier = nn.Linear(self.d_model, self.num_classes)

        self.work_num = int(cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS)

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            visual_inputs_long = visual_inputs[:, :self.long_memory_num_samples]
            motion_inputs_long = motion_inputs[:, :self.long_memory_num_samples]
            extend_index = torch.arange(1,self.num_history+1) * self.history_rate - 1
            
            visual_inputs_extend = visual_inputs[:,self.long_memory_num_samples:][:,extend_index,:]
            motion_inputs_extend = motion_inputs[:,self.long_memory_num_samples:][:,extend_index,:]
            visual_inputs_long = torch.cat((visual_inputs_long, visual_inputs_extend), dim=1)
            motion_inputs_long = torch.cat((motion_inputs_long, motion_inputs_extend), dim=1)
            attn_map = motion_inputs_long

            long_memories = self.feature_head_long(
                visual_inputs_long,
                motion_inputs_long,
            ).transpose(0, 1)
            long_memories_extend = long_memories[self.long_memory_num_samples:,:,:]
            long_memories = long_memories[:self.long_memory_num_samples,:,:] 
            
            memory, memory_cached = self.history_encoder(long_memories, long_memories_extend, memory_key_padding_mask)

            if self.oas:
                unzip_history = self.his_decoder(memory.transpose(0,1), memory_cached)
                history_score = self.classifier(unzip_history)
            else:
                history_score = None
        else:
            memory = None
            memory_cached = None
            history_score = None
        
        if self.work_enabled:
            visual_inputs_work = visual_inputs[:, self.long_memory_num_samples:]
            motion_inputs_work = motion_inputs[:, self.long_memory_num_samples:]

            work_memories = self.feature_head_work(
                visual_inputs_work,
                motion_inputs_work,
            ).transpose(0, 1)

            if self.long_enabled:
                memories = torch.cat((memory, work_memories), dim=0)
            else:
                memories = work_memories
        else:
            memories = memory
        
        queries = work_memories

        if self.long_enabled:
            dec_output = self.dec_modules(
                queries, memories,
                memory_key_padding_mask=memory_key_padding_mask)

            if self.cascade:
                if self.cascade_shared:
                    cas_output = self.cas_module(dec_output.transpose(0,1))
                else:
                    cas_output = self.cas_module(dec_output.transpose(0,1))
                    if self.oas_cascade:
                        cas_output_h = self.cas_module_h(memory_cached.transpose(0,1))
                        cas_output = torch.cat([dec_output.transpose(0,1).unsqueeze(0), cas_output], dim=0)
                        cas_output = torch.cat([cas_output, cas_output_h.unsqueeze(0)], dim=0)
                    else:
                        cas_output = torch.cat([dec_output.transpose(0,1).unsqueeze(0), cas_output], dim=0)

                scores = []
                for cas_out in cas_output:
                    scores.append(self.classifier(cas_out))
                det_score = torch.stack(scores, dim=0)
            else:
                det_score = self.classifier(dec_output.transpose(0,1))
        else:
            det_score = self.classifier(work_memories.transpose(0,1))
            history_score = None

        return (det_score, history_score)

    def batch_inference(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            visual_inputs_long = visual_inputs[:, :self.long_memory_num_samples]
            motion_inputs_long = motion_inputs[:, :self.long_memory_num_samples]
            extend_index = torch.arange(1,self.num_history+1) * self.history_rate - 1
            
            visual_inputs_extend = visual_inputs[:,self.long_memory_num_samples:][:,extend_index,:]
            motion_inputs_extend = motion_inputs[:,self.long_memory_num_samples:][:,extend_index,:]
            visual_inputs_long = torch.cat((visual_inputs_long, visual_inputs_extend), dim=1)
            motion_inputs_long = torch.cat((motion_inputs_long, motion_inputs_extend), dim=1)

            long_memories = self.feature_head_long(
                visual_inputs_long,
                motion_inputs_long,
            ).transpose(0, 1)
            long_memories_extend = long_memories[self.long_memory_num_samples:,:,:]
            long_memories = long_memories[:self.long_memory_num_samples,:,:] 
            
            memory, memory_cached = self.history_encoder(long_memories, long_memories_extend, memory_key_padding_mask)

            if self.oas:
                unzip_history = self.his_decoder(memory.transpose(0,1), memory_cached)
                history_score = self.classifier(unzip_history)
            else:
                history_score = None
        else:
            memory = None
            memory_cached = None
            history_score = None
        
        if self.work_enabled:
            visual_inputs_work = visual_inputs[:, self.long_memory_num_samples:]
            motion_inputs_work = motion_inputs[:, self.long_memory_num_samples:]

            work_memories = self.feature_head_work(
                visual_inputs_work,
                motion_inputs_work,
            ).transpose(0, 1)

            if self.long_enabled:
                memories = torch.cat((memory, work_memories), dim=0)
            else:
                memories = work_memories
        else:
            memories = memory
        
        queries = work_memories

        if self.long_enabled:
            dec_output = self.dec_modules(
                queries, memories,
                memory_key_padding_mask=memory_key_padding_mask)

            if self.cascade:
                if self.cascade_shared:
                    cas_output = self.cas_module(dec_output.transpose(0,1))
                else:
                    cas_output = self.cas_module(dec_output.transpose(0,1))
                    if self.oas_cascade:
                        cas_output_h = self.cas_module_h(memory_cached.transpose(0,1))
                        cas_output = torch.cat([dec_output.transpose(0,1).unsqueeze(0), cas_output], dim=0)
                        cas_output = torch.cat([cas_output, cas_output_h.unsqueeze(0)], dim=0)
                    else:
                        cas_output = torch.cat([dec_output.transpose(0,1).unsqueeze(0), cas_output], dim=0)

                scores = []
                for cas_out in cas_output:
                    scores.append(self.classifier(cas_out))
                det_score = torch.stack(scores, dim=0)
            else:
                det_score = self.classifier(dec_output.transpose(0,1))
        else:
            det_score = self.classifier(work_memories.transpose(0,1))
            history_score = None

        return (det_score, history_score)