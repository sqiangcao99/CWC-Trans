import torch
import torch.nn as nn
from .segs import ASTransformer, RMTSM, RLinear 
from .segs import ASTransformer, RMTSM, RLinear

class HistoryDecoder(nn.Module):

    def __init__(
                    self, 
                    output_tokens=[32, 128, 512, 512],
                    num_layers=[4, 4, 4, 4], 
                    embed_dim = [1024, 1024, 1024, 1024], 
                    num_heads = [4,4,4,4],
                    window_size = [4, 8, 32, 64],
                    num_patches = [16, 32, 128, 512],
                    num_tsm = [8,8,8,8],
                    tsm_dropout = 0.1, 
                    islocal= False,
                    iscut = False,
                    shcut_dict = None): 
        super().__init__()
        self.depth = len(output_tokens) - 1
        
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.is_cut = iscut
        
        for i_layer in range(self.depth):
            attn_layer = ASTransformer(
                                        embed_dim=embed_dim[i_layer], depths=num_layers[i_layer], num_heads=num_heads[i_layer],
                                        window_size=window_size[i_layer],num_patches=num_patches[i_layer],
                                        mlp_ratio=4.,qkv_bias=True, qk_scale=None,
                                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                        iscut=iscut)
    
            if i_layer == 0 and islocal:
                down_layer = RLinear(
                                input_dim = int(embed_dim[i_layer]), 
                                hidden_dim = int(embed_dim[i_layer] * 1.5),
                                output_num = output_tokens[i_layer],
                )
            else:                                        
                down_layer = RMTSM(
                                    input_dim=int(embed_dim[i_layer]), 
                                    hidden_dim = int(embed_dim[i_layer]) // 2,
                                    output_num = output_tokens[i_layer],
                                    num_heads = num_tsm[i_layer],
                                    dropout = tsm_dropout,
                                    islocal=islocal
                                    )
            self.layers.append(attn_layer)
            self.downsample_layers.append(down_layer)
        
        self.global_ops = attn_layer = ASTransformer(
                                        embed_dim=embed_dim[-1], depths=num_layers[-1], num_heads=num_heads[-1],
                                        window_size=window_size[-1],num_patches=num_patches[-1],
                                        mlp_ratio=4.,qkv_bias=True, qk_scale=None,
                                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, iscut=iscut)
        
        if self.is_cut:
            self.shcut_module = nn.ModuleList()
            shcut_emb_dim = shcut_dict['DENSE']['EMB_DIM']
            for dim in shcut_dict['DENSE']['MERGE_DIM']:
                layer = nn.Sequential(
                    nn.Linear(dim, shcut_emb_dim),
                    nn.ReLU()
                )
                    
                self.shcut_module.append(layer)
        
    def forward(self, x, shortcuts=None):   

        shortcuts.reverse()
        for i_layer in range(self.depth):
            if self.is_cut and i_layer >= 1:
                shortcut = shortcuts[i_layer-1]
                x = torch.cat((shortcut, x), dim=-1)
                x = self.shcut_module[i_layer-1](x)

            x = self.layers[i_layer](x)
            x = self.downsample_layers[i_layer](x)
        
        if self.is_cut:
            shortcut = shortcuts[-1]
            x = torch.cat((shortcut, x), dim=-1)
            x = self.shcut_module[-1](x)
        
        x = self.global_ops(x)
        return x


