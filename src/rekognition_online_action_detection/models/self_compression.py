# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn

from .tsw import SwinTransformer
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .feature_head import build_feature_head

class SelfCompression(nn.Module):
    
    def __init__(self, cfg):
        super(SelfCompression, self).__init__()
        
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        
        self.current_extend = int(cfg.DATA.FPS * cfg.MODEL.LSTR.WORK_MEMORY_SECONDS)
        self.history_extend = int(self.long_memory_num_samples + np.around(self.current_extend / cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE)) - 1 
        self.history_length = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.history_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.num_history = int(self.current_extend / cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE) 

        self.window_num = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS // cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE
        self.window_size = cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE 
        self.encoder = SwinTransformer(
                            cfg,
                            embed_dim=cfg.MODEL.SWIN.EMBED_DIM, 
                            depths=cfg.MODEL.SWIN.DEPTHS, 
                            num_heads=cfg.MODEL.SWIN.NUM_HEADS, 
                            window_size=cfg.MODEL.SWIN.WINDOW_SIZE, 
                            out_size = cfg.MODEL.SWIN.OUT_SIZE, 
                            mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=cfg.MODEL.SWIN.QKV_BIAS, 
                            qk_scale=cfg.MODEL.SWIN.QK_SCALE,
                            drop_rate=cfg.MODEL.SWIN.DROP_RATE, 
                            attn_drop_rate=cfg.MODEL.SWIN.ATTN_DROP_RATE, 
                            drop_path_start=cfg.MODEL.SWIN.DROP_PATH_START,
                            drop_path_rate=cfg.MODEL.SWIN.DROP_PATH_RATE,
                            norm_layer=nn.LayerNorm, 
                            patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
        )
        
        self.share_weights = True
        if not self.share_weights:
            self.rest_encoder = SwinTransformer(
                                embed_dim=cfg.MODEL.SWIN.EMBED_DIM, 
                                depths=cfg.MODEL.SWIN.DEPTHS, 
                                num_heads=cfg.MODEL.SWIN.NUM_HEADS, 
                                num_patches=cfg.MODEL.SWIN.NUM_REST_PATCHES, 
                                mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=cfg.MODEL.SWIN.QKV_BIAS, 
                                qk_scale=cfg.MODEL.SWIN.QK_SCALE,
                                drop_rate=cfg.MODEL.SWIN.DROP_RATE, 
                                attn_drop_rate=cfg.MODEL.SWIN.ATTN_DROP_RATE, 
                                drop_path_rate=cfg.MODEL.SWIN.DROP_PATH_RATE,
                                norm_layer=nn.LayerNorm, 
                                patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
            )
        
        self.history_process = cfg.MODEL.LSTR.HISTORY.PROCESS.ENC
        self.history_process_dim = cfg.MODEL.SWIN.EMBED_DIM * 2**len(cfg.MODEL.SWIN.DEPTHS)
        self.history_process_feedforward = cfg.MODEL.SWIN.EMBED_DIM * 2**len(cfg.MODEL.SWIN.DEPTHS)
        self.history_process_num_heads = cfg.MODEL.LSTR.HISTORY.PROCESS.NUM_HEADS
        self.process_qkv = [cfg.MODEL.SWIN.EMBED_DIM * 2 ** len(cfg.MODEL.SWIN.DEPTHS)] * 3 
        
        self.globals = cfg.MODEL.SWIN.GLOBAL
        
        if self.globals:
            process_layer = tr.TransformerEncoderLayer(
                d_model = self.history_process_dim, 
                nhead = self.history_process_num_heads, 
                dim_feedforward = self.history_process_feedforward,
                dropout = self.dropout, 
                activation = self.activation)

            self.process_module = tr.TransformerEncoder(
                encoder_layer = process_layer, 
                num_layers = self.history_process[1], 
                norm = tr.layer_norm(self.history_process_feedforward, self.history_process[2])
            )
        
        self.pooling = 'mean'  
        self.pos_emb = tr.PositionalEncoding(cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES_LONG, self.dropout).pe
        self.pos_emb_update = self.pos_emb[:cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE*2, :,:]
        self.pos_emb_fixed = self.pos_emb[cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE:cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE*2, :,:]
        
        self.abs_pos_emb = tr.PositionalEncoding(cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES, self.dropout).pe
        self.abs_pos_emb_offset =  self.abs_pos_emb[::cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE,:,:] 

        self.window_size = cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE
        self.window_num = int(cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES // cfg.MODEL.LSTR.HISTORY.WINDOW.SIZE)

        self.oproj = cfg.MODEL.LSTR.HISTORY.WINDOW.OPROJ
        if self.oproj:
            self.oproj_layer = nn.Linear(self.history_process_dim, cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES)
    def forward(self, long_memories, memories_extend, memory_key_padding_mask=None):
        long_memories = long_memories.view(self.window_num, self.window_size, long_memories.size(1), long_memories.size(2))
        mask_extend = torch.zeros(memory_key_padding_mask.shape[0], self.num_history).to(memory_key_padding_mask.device)
        memory_key_padding_mask_window = memory_key_padding_mask.view(memory_key_padding_mask.size(0), self.window_num, self.window_size)
        old_memory_key_padding_mask_window = memory_key_padding_mask_window[:,0]
        fixed_memory_key_padding_mask_window = memory_key_padding_mask_window[:,1:]
        merged_memory_key_padding_mask_window = torch.cat((old_memory_key_padding_mask_window, mask_extend),dim=1)
        
        old_window_memories = long_memories[0]
        fixed_window_memories = long_memories[1:]
        new_window_memories = memories_extend
   
        merged_window_memories = torch.cat((old_window_memories, new_window_memories),dim=0) 
        assert merged_window_memories.shape[0] <= self.window_size * 2
        nw, _, bsz, dim = fixed_window_memories.shape 
        fixed_window_memories = fixed_window_memories.permute(1, 0, 2, 3).contiguous().view(self.window_size, -1, dim)
        fixed_memory_key_padding_mask_window = fixed_memory_key_padding_mask_window.view(-1, self.window_size)
        
        fixed_window_memories = fixed_window_memories.transpose(0,1)
        fixed_window_memories, cached_outputs_fixed = self.encoder(fixed_window_memories, fixed_memory_key_padding_mask_window)

        if len(fixed_window_memories.shape) > 2:
            if self.pooling == 'mean':
                fixed_window_memories = torch.mean(fixed_window_memories, dim=1)
            elif self.pooling == 'max':
                fixed_window_memories = torch.max(fixed_window_memories, dim=1)[0]
        fixed_window_memories = fixed_window_memories.view(bsz, -1, fixed_window_memories.size(-1))
        merged_window_memories = merged_window_memories.transpose(0,1)
        merged_window_memories = [merged_window_memories[:,i:i+self.window_size,:] for i in range(self.num_history)]
        merged_window_memories = torch.cat(merged_window_memories, dim=0)

        merged_memory_key_padding_mask_window = [merged_memory_key_padding_mask_window[:,i:i+self.window_size] for i in range(self.num_history)]
        merged_memory_key_padding_mask_window = torch.cat(merged_memory_key_padding_mask_window, dim=0)
        merged_window_memories, cached_outputs_merged = self.encoder(merged_window_memories, merged_memory_key_padding_mask_window)
        if len(merged_window_memories.shape) > 2:
            if self.pooling == 'mean':
                merged_window_memories = torch.mean(merged_window_memories, dim=1)
            elif self.pooling == 'max':
                merged_window_memories = torch.max(merged_window_memories, dim=1)[0]
        nh =  memories_extend.shape[0]
        cached_outputs = []
        for ly in range(len(cached_outputs_fixed)):
            cached_output_fixed = cached_outputs_fixed[ly]
            cached_output_merged = cached_outputs_merged[ly]
            
            cached_output_fixed = cached_output_fixed.view(bsz, -1, self.window_size, cached_output_fixed.size(-1))
            cached_output_fixed = cached_output_fixed.unsqueeze(1).expand(-1, nh, -1, -1, -1)
            cached_output_merged = cached_output_merged.view(bsz, nh, 1, self.window_size, cached_output_merged.size(-1))
            
            cached_output = torch.cat((cached_output_merged, cached_output_fixed), dim = 2)
            cached_output = cached_output.permute(0, 1, 2, 3, 4).contiguous().view(bsz * nh, -1, cached_output.size(-1))
            cached_outputs.append(cached_output)

        merged_window_memories = merged_window_memories.view(bsz, -1, merged_window_memories.size(-1))
        merged_window_memories = merged_window_memories.unsqueeze(2).expand(-1, -1, self.window_num, -1)
        fixed_window_memories = fixed_window_memories.unsqueeze(1).expand(-1, nh, -1, -1)
        
        window_memories = torch.cat((merged_window_memories, fixed_window_memories), dim = 2)
        current_history_window = window_memories
        current_history_window = current_history_window.permute(2, 0, 1, 3).contiguous().view(current_history_window.size(2), -1, current_history_window.size(-1))
        if self.globals:
            current_history_window = self.process_module(current_history_window)
        
        if self.oproj:
            current_history_window = self.oproj_layer(current_history_window)
        return current_history_window, cached_outputs

    def batch_inference(self, long_memories, memory_key_padding_mask=None):
        nw = long_memories.shape[0]  //self.window_size
        long_memories = long_memories.view(nw, self.window_size, long_memories.size(1), long_memories.size(2))
        long_memories = long_memories.permute(1, 2, 0, 3).contiguous().view(-1, nw, long_memories.size(-1))
        
        memory_key_padding_mask_window = memory_key_padding_mask.view(memory_key_padding_mask.size(0), -1, self.window_size).permute(0, 2, 1).contiguous().view(-1, self.window_size)     
        long_memories, cached_outputs_fixed = self.encoder(long_memories, memory_key_padding_mask_window)
        if len(long_memories.shape) > 2:
            if self.pooling == 'mean':
                long_memories = torch.mean(long_memories, dim=1)
            elif self.pooling == 'max':
                long_memories = torch.max(long_memories, dim=1)[0]
        long_memories = long_memories.view(nw, -1, long_memories.size(-1))
        
        if self.globals:
            long_memories = self.process_module(long_memories)
 

        if self.oproj:
            current_history_window = self.oproj_layer(current_history_window)
        return long_memories

    def window_compression(self, long_memories, memory_key_padding_mask=None):   
        long_memories, cached_outputs_fixed = self.encoder(long_memories, memory_key_padding_mask) 
        if len(long_memories.shape) > 2:
            if self.pooling == 'mean':
                long_memories = torch.mean(long_memories, dim=1)
            elif self.pooling == 'max':
                long_memories = torch.max(long_memories, dim=1)[0]
        long_memories = long_memories.view(-1, 1, long_memories.size(-1))
        return long_memories

