# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class WindowAttention(nn.Module):
    def __init__(self, dim, seq_len, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * seq_len - 1) , num_heads)) 
        
 
       
        coords = torch.arange(self.seq_len)
        relative_coords = coords[:, None] - coords[None, :] 
        relative_coords += self.seq_len - 1 
        relative_position_index = relative_coords
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.seq_len, self.seq_len, -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
           
            attn = attn + mask.unsqueeze(1).unsqueeze(1)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, 
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
 
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, seq_len=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, mask=None):
        shortcut = x
        x = self.norm1(x) 
        x = self.attn(x, mask=mask)
       
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x
class MTSMV1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_num, num_heads=2,drop_rate=0.1, attn_drop=0.1, out_drop=0.5):
        super(MTSMV1, self).__init__()
        self.nh = num_heads
        self.input_size = input_dim 
        self.hidden_size = hidden_dim 
        self.output_num = output_num
        self.to_qkv = nn.Linear(input_dim, input_dim)
       
        self.pre_norm = nn.LayerNorm(input_dim)  
        self.prob_map = nn.Sequential(
            nn.Linear(input_dim // self.nh, hidden_dim // self.nh),
            nn.ReLU(),
            nn.Linear(hidden_dim // self.nh, self.output_num), 
        )
       
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim , input_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(input_dim * 2, input_dim * 2),
        )
        self.attn_dropout = nn.Dropout(attn_drop) 
        self.out_dropout = nn.Dropout(out_drop)
        
        kernel_skip = [4 + 1]
        stride_skip = 4
        padding_skip = [int(skip // 2) for skip in kernel_skip]
       
        self.pool_skip = nn.AvgPool1d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
    def forward(self, input, mask=None):
        if mask != None:
            mask = mask.float().masked_fill(mask == 0, float(1)).masked_fill(mask < 0, float(0.0))
            input = input * mask.unsqueeze(-1)
        else:
            input = input
        input = self.pre_norm(input) 
        
        x_res = self.pool_skip(input.transpose(1,2)).transpose(1,2)
        qkv = self.to_qkv(input)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.nh, -1).permute(0, 2, 1, 3).contiguous().view(-1, qkv.size(1), qkv.size(-1) // self.nh)
        self_prob = self.prob_map(qkv).softmax(dim=-2)
        self_prob = self.attn_dropout(self_prob)
        output = torch.einsum('b w o, b w d-> b o d', (self_prob, qkv))
        output = output.view(-1, self.nh, output.size(1), output.size(-1)).permute(0, 2, 1, 3).contiguous().view(output.size(0) // self.nh, output.size(1), -1)
        
        output = x_res + self.out_dropout(output)
        
        output = self.ffn_norm(output)
        output = self.ffn(output)
        output = self.out_dropout(output)
        return output
class BasicLayer(nn.Module):
    def __init__(self, dim,  depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
       
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, 
                                 num_heads=num_heads, 
                                 window_size = window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
       
        if downsample is not None:
           
            self.downsample = downsample
        else:
            self.downsample = None
    def forward(self, x, mask = None):
        
        for blk in self.blocks:
            x = blk(x, mask=mask)
        if self.downsample is not None:
            x = self.downsample(x,mask) 
        return x
class SwinTransformer(nn.Module):
    def __init__(self, cfg, embed_dim=256, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size = [32, 16, 8, 4], out_size = [16, 8,4,1],
                mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, drop_path_start = 0.,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                **kwargs):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.out_size = out_size 
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.mlp_ratio = mlp_ratio
       
        self.pos_drop = nn.Dropout(p=drop_rate)
       
        dpr = [x.item() for x in torch.linspace(drop_path_start, drop_path_rate, sum(depths))] 
       
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size = self.window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=MTSMV1(  
                                                input_dim=int(embed_dim * 2 ** i_layer), 
                                                hidden_dim = int(embed_dim * 2 ** i_layer) // 2,
                                                output_num = max(1, self.out_size[i_layer]),
                                                num_heads = 8,
                                                drop_rate = cfg.MODEL.SWIN.TSM.DROPOUT, 
                                                attn_drop = cfg.MODEL.SWIN.TSM.ATTN_DROPOUT, 
                                                out_drop = cfg.MODEL.SWIN.TSM.OUT_DROPOUT, 
                                                )
                                )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def forward_features(self, x, key_padding_mask = None):
        x = self.pos_drop(x)
        layer_output = []
        layer_output.append(x)
        
        for layer in self.layers:
            if key_padding_mask != None:
                valid_mask = (key_padding_mask == 0).sum(-1) > 0
                x = x[valid_mask] 
                valid_window_mask = key_padding_mask[valid_mask]
                x = layer(x, mask = valid_window_mask)
                x_ = torch.zeros(valid_mask.shape[0], *x.shape[1:]).to(x.device)
                x_[valid_mask] = x
                x = x_
            else:
                x = layer(x)
            
            layer_output.append(x)
            key_padding_mask = None
        x = self.norm(x)
        return x, layer_output
    def forward(self, x, key_padding_mask = None):
        x, layer_output = self.forward_features(x, key_padding_mask=key_padding_mask)
        return x, layer_output
