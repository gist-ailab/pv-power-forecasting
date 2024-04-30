__all__ = ['PatchCDTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchCDTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = CDTSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                      n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                      attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    
    def forward(self, src_domain, tgt_domain):                                              # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            src_domain = src_domain.permute(0,2,1)
            src_domain = self.revin_layer(src_domain, 'norm')
            src_domain = src_domain.permute(0,2,1)
            tgt_domain = tgt_domain.permute(0,2,1)
            tgt_domain = self.revin_layer(tgt_domain, 'norm')
            tgt_domain = tgt_domain.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            src_domain = self.padding_patch_layer(src_domain)
            tgt_domain = self.padding_patch_layer(tgt_domain)
        src_domain = src_domain.unfold(dimension=-1, size=self.patch_len, step=self.stride) # z: [bs x nvars x patch_num x patch_len]
        tgt_domain = tgt_domain.unfold(dimension=-1, size=self.patch_len, step=self.stride) # z: [bs x nvars x patch_num x patch_len]
        src_domain = src_domain.permute(0,1,3,2)                                            # z: [bs x nvars x patch_len x patch_num]
        tgt_domain = tgt_domain.permute(0,1,3,2)                                            # z: [bs x nvars x patch_len x patch_num]
        
        # model
        src_z, tgt_z, cross_z = self.backbone(src_domain, tgt_domain)       # z: [bs x nvars x d_model x patch_num]
        src_output = self.head(src_z)                                       # output: [bs x nvars x target_window]
        tgt_output = self.head(tgt_z)                                       # output: [bs x nvars x target_window]
        
        
        # denorm
        if self.revin: 
            src_output = src_output.permute(0,2,1)
            src_output = self.revin_layer(src_output, 'denorm')
            src_output = src_output.permute(0,2,1)
            
            tgt_output = tgt_output.permute(0,2,1)
            tgt_output = self.revin_layer(tgt_output, 'denorm')
            tgt_output = tgt_output.permute(0,2,1)          
        return src_output, tgt_output, tgt_z, cross_z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z
    

class CDTSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = CDTSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, src, tgt) -> Tensor:                                      # input: [bs x nvars x patch_len x patch_num]
        
        n_vars = src.shape[1]
        if src.shape[1] != tgt.shape[1]: raise ValueError('source and target must have the same number of variables')
        # Input encoding
        src = src.permute(0,1,3,2)                                             # src: [bs x nvars x patch_num x patch_len]
        src = self.W_P(src)                                                    # src: [bs x nvars x patch_num x d_model]
        tgt = tgt.permute(0,1,3,2)                                             # tgt: [bs x nvars x patch_num x patch_len]
        tgt = self.W_P(tgt)                                                    # tgt: [bs x nvars x patch_num x d_model]
        
        src_u = torch.reshape(src, (src.shape[0]*src.shape[1],src.shape[2],src.shape[3]))   # u: [bs * nvars x patch_num x d_model]
        src_u = self.dropout(src_u + self.W_pos)                                            # u: [bs * nvars x patch_num x d_model]
        tgt_u = torch.reshape(tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2],tgt.shape[3]))   # u: [bs * nvars x patch_num x d_model]
        tgt_u = self.dropout(tgt_u + self.W_pos)                                            # u: [bs * nvars x patch_num x d_model]
        
        # Encoder
        src_z, tgt_z, cross_z = self.encoder(src_u, tgt_u)                                  # z: [bs * nvars x patch_num x d_model]
        src_z = torch.reshape(src_z, (-1,n_vars,src_z.shape[-2],src_z.shape[-1]))           # z: [bs x nvars x patch_num x d_model]
        src_z = src_z.permute(0,1,3,2)                                                      # z: [bs x nvars x d_model x patch_num]
        tgt_z = torch.reshape(tgt_z, (-1,n_vars,tgt_z.shape[-2],tgt_z.shape[-1]))           # z: [bs x nvars x patch_num x d_model]
        tgt_z = tgt_z.permute(0,1,3,2)                                                      # z: [bs x nvars x d_model x patch_num]
        cross_z = torch.reshape(cross_z, (-1,n_vars,cross_z.shape[-2],cross_z.shape[-1]))   # z: [bs x nvars x patch_num x d_model]
        cross_z = cross_z.permute(0,1,3,2)                                                  # z: [bs x nvars x d_model x patch_num]
        
        return src_z, tgt_z, cross_z
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


# Cell
class CDTSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        # self.layers = nn.ModuleList([
        #     nn.ModuleList([
        #                 TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
        #                                 attn_dropout=attn_dropout, dropout=dropout,
        #                                 activation=activation, res_attention=res_attention,
        #                                 pre_norm=pre_norm, store_attn=store_attn),
        #                 TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
        #                                 attn_dropout=attn_dropout, dropout=dropout,
        #                                 activation=activation, res_attention=res_attention,
        #                                 pre_norm=pre_norm, store_attn=store_attn),
        #                 CDTSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
        #                                 attn_dropout=attn_dropout, dropout=dropout,
        #                                 activation=activation, res_attention=res_attention,
        #                                 pre_norm=pre_norm, store_attn=store_attn)
        #                 ])  # TODO: weight sharing 해야하므로 이렇게 되어있으면 안 된다.
        # ])
        
        self.layers = nn.ModuleList([CDTSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                       attn_dropout=attn_dropout, dropout=dropout,
                                                       activation=activation, res_attention=res_attention,
                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])

        self.res_attention = res_attention

    def forward(self, src_s:Tensor, src_t:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output_s = src_s
        output_t = src_t
        output_cd = torch.zeros_like(src_s)
        scores_s = None
        scores_t = None
        scores_cd = None

        # if self.res_attention:
        #     for src_layer, tgt_layer, cross_layer in self.layers:
        #         output_s, scores_s = src_layer(output_s, prev=scores_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        #         output_t, scores_t = tgt_layer(output_t, prev=scores_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        #         output_cross, scores_cross = cross_layer(output_s, output_t, prev=scores_cross, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # else:
        #     for src_layer, tgt_layer, cross_layer in self.layers:
        #         output_s = src_layer(output_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        #         output_t = tgt_layer(output_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        #         output_cross = cross_layer(output_s, output_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                
        if self.res_attention:
            for cross_layer in self.layers:
                output_s, output_t, output_cd, scores_s, scores_t, scores_cd = cross_layer(output_s, output_t, output_cd,
                                                                                           prev_s=scores_s, prev_t=scores_t, prev_cd=scores_cd,
                                                                                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            for cross_layer in self.layers:
                output_s, output_t, output_cd = cross_layer(output_s, output_t, output_cd,
                                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        return output_s, output_t, output_cd

        
        



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
            
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

# CDTSTEncoderLayer에선 source domain의 Q와 target domain의 K, V를 받아서 attention을 수행한다.
class CDTSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src_s:Tensor, src_t:Tensor, src_cd:Tensor,
                prev_s:Optional[Tensor]=None, prev_t:Optional[Tensor]=None, prev_cd:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # TODO: encoder의 layer 번호를 가져와서 첫 번째 layer인 경우, index를 _MultiheadAttention에 전달해준다.
        # Multi-Head attention sublayer
        if self.pre_norm:
            src_s = self.norm_attn(src_t)
            src_t = self.norm_attn(src_t)
            src_cd = self.norm_attn(src_cd)
            
        for i in range(3):
            if i == 0:
                if self.res_attention:
                    output_s, q_s, _, _, scores_s = self.source_target_encoder_layer(src_s, src_s, src_s, src_s, prev_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                else:
                    output_s, q_s, _, _ = self.source_target_encoder_layer(src_s, src_s, src_s, src_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                    
            elif i ==1:
                if self.res_attention:
                    output_t, _, v_t, k_t, scores_t = self.source_target_encoder_layer(src_t, src_t, src_t, src_t, prev_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                else:
                    output_t, _, v_t, k_t = self.source_target_encoder_layer(src_t, src_t, src_t, src_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            
            else:
                if self.res_attention:
                    output_cd, _, _, _, scores_cd = self.source_target_encoder_layer(src_cd, q_s, v_t, k_t, prev_cd, key_padding_mask=key_padding_mask, attn_mask=attn_mask, cross_domain=True)
                else:
                    output_cd, _, _, _ = self.source_target_encoder_layer(src_cd, q_s, v_t, k_t, key_padding_mask=key_padding_mask, attn_mask=attn_mask, cross_domain=True)

        return output_s, output_t, output_cd, scores_s, scores_t, scores_cd      

        
    def source_target_encoder_layer(self, src:Tensor, Q:Tensor, K:Tensor, V:Tensor, prev:Optional[Tensor]=None,
                                    key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, cross_domain=False) -> Tensor:
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, q, k, v, scores = self.self_attn(Q, K, V, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask, cross_domain=cross_domain)
        else:
            src2, attn, q, k, v = self.self_attn(Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask, cross_domain=cross_domain)
        if self.store_attn:
            self.attn = attn
            
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, q, k, v, scores
        else:
            return src, q, k, v



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, cross_domain=False):
        # TODO: layer index를 받아와서 cross_domain에 대한 hidden feature를 0으로 할지 말지 결정해준다.
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        if cross_domain:
            q_s = Q
            k_s = K
            v_s = V
        else:
            # Linear (+ split in multiple heads)
            q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
            k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
            v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, q_s, k_s, v_s, attn_scores
        else: return output, attn_weights, q_s, k_s, v_s


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

