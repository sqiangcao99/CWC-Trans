# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer as tr
class WindowDotProduction(nn.Module):

    def __init__(self, window_size = 32, dropout=0.0):
        super(WindowDotProduction, self).__init__()
        self.dropout = dropout
        self.window_size = window_size

    def forward(self, q, k, v, attn_mask=None):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)

        attn_output_weights = attn_output_weights.unfold(-1, self.window_size, 1)
        v = v.unfold(-2, self.window_size, 1).permute(0,1,3,2)
        attn_output = torch.einsum('bths, bhse-> bthe', attn_output_weights, v)
        return attn_output


class WindowMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=32, dropout=0.0, bias=True, qdim=None, kdim=None):
        super(WindowMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qdim = qdim
        self.kdim = kdim
        self.window_size = window_size

        self.in_proj_weight_q = nn.Parameter(torch.empty(qdim, qdim))
        self.in_proj_weight_kv = nn.Parameter(torch.empty(2*kdim, qdim)) 
        
        if bias:
            self.in_proj_bias_q = nn.Parameter(torch.empty(qdim))
            self.in_proj_bias_kv = nn.Parameter(torch.empty(2*qdim))
        else:
            self.register_parameter('in_proj_bias_q', None)
            self.register_parameter('in_proj_bias_kv', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        nn.init.xavier_uniform_(self.in_proj_weight_q)
        nn.init.xavier_uniform_(self.in_proj_weight_kv)

        if self.in_proj_bias_q is not None:
            nn.init.constant_(self.in_proj_bias_q, 0.)
            nn.init.constant_(self.in_proj_bias_kv, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        self.dotproductattention = WindowDotProduction(window_size=window_size, dropout=dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5
        q_w = self.in_proj_weight_q
        q_b = self.in_proj_bias_q
        if q_b is not None:
            q_b = self.in_proj_bias_q
        
        q = F.linear(q, q_w, q_b)
        
        k_w = self.in_proj_weight_kv[:self.kdim, :]
        k_b = self.in_proj_bias_kv
        if k_b is not None:
            k_b = self.in_proj_bias_kv[:self.qdim]

        k = F.linear(k, k_w.T, k_b)

        v_w = self.in_proj_weight_kv[self.kdim:2*self.kdim, :]
        v_b = self.in_proj_bias_kv
        if v_b is not None:
            v_b = self.in_proj_bias_kv[self.qdim:2*self.qdim]

        v = F.linear(v, v_w.T, v_b)  

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
            # pdb.set_trace()
        else:
            mask = None
        
        attn_output = self.dotproductattention(q, k, v, attn_mask = mask)
        nt, nh = attn_output.shape[1:3]
        attn_output = attn_output.permute(2, 1,0,3).contiguous().view(nh, tsz, bsz, self.embed_dim)
        return self.out_proj(attn_output), None

class WindowCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, num_tokens, window_size=32, dim_feedforward=2048, dropout=0.1, activation='relu', qdim=None, kdim=None):
        super(WindowCrossAttentionLayer, self).__init__()
        self.window_size = window_size
        self.self_attn = tr.MultiheadAttention(d_model, nhead, qdim=qdim, kdim=qdim ,dropout=dropout)
        self.multihead_attn = WindowMultiheadAttention(d_model, nhead, window_size=self.window_size, qdim=qdim , kdim=kdim ,dropout=dropout) 
        assert d_model == qdim
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward) 

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        ## Generate table index
        self.num_tokens = num_tokens
        self.relative_pos = nn.Parameter(torch.zeros((nhead, 1,num_tokens)))
        self._init_parameters()
    
    def _init_parameters(self):
        with torch.no_grad():
            self.relative_pos.normal_(0.0, 0.02).clamp_(-2.0, 2.0) 

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(WindowCrossAttentionLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]                     
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class WindowCrossAttention(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(WindowCrossAttention, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))