__all__ = ['PatchCDTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchCDTST_backbone import PatchCDTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchCDTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchCDTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchCDTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, src_domain, tgt_domain):      # src_domain, tgt_domain: [Batch, Input length, Channel]
        if self.decomposition:
            src_res_init, src_trend_init = self.decomp_module(src_domain)
            src_res_init, src_trend_init = src_res_init.permute(0,2,1), src_trend_init.permute(0,2,1)  # src_domain: [Batch, Channel, Input length]
            tgt_res_init, tgt_trend_init = self.decomp_module(tgt_domain)
            tgt_res_init, tgt_trend_init = tgt_res_init.permute(0,2,1), tgt_trend_init.permute(0,2,1)  # tgt_domain: [Batch, Channel, Input length]
            
            src_res, tgt_res, tgt_res_feat, cross_res_feat = self.model_res(src_res_init, tgt_res_init)
            src_trend, tgt_trend, tgt_trend_feat, cross_trend_feat = self.model_trend(src_trend_init, tgt_res_init)
            
            src_domain = src_res + src_trend
            tgt_domain = tgt_res + tgt_trend
            tgt_feat = tgt_res_feat + tgt_trend_feat
            cross_feat = cross_res_feat + cross_trend_feat
            # TODO: 위 방식이 정확히 어떤 의미를 가질 지 잘 모르겠음. 우선 코드 작동하는데 문제 없도록만 해둠.            
            
            src_domain = src_domain.permute(0,2,1)      # src_domain: [Batch, Input length, Channel]
            tgt_domain = tgt_domain.permute(0,2,1)      # tgt_domain: [Batch, Input length, Channel]
        else:
            src_domain = src_domain.permute(0,2,1)      # src_domain: [Batch, Channel, Input length]
            tgt_domain = tgt_domain.permute(0,2,1)      # tgt_domain: [Batch, Channel, Input length]
            src_domain, tgt_domain, tgt_feat, cross_feat = self.model(src_domain, tgt_domain)
            src_domain = src_domain.permute(0,2,1)      # src_domain: [Batch, Input length, Channel]
            tgt_domain = tgt_domain.permute(0,2,1)      # tgt_domain: [Batch, Input length, Channel]
            tgt_feat = tgt_feat.permute(0,2,1)          # tgt_feat: [Batch, Input length, Channel]
            cross_feat = cross_feat.permute(0,2,1)      # cross_feat: [Batch, Input length, Channel]
            
        return src_domain, tgt_domain, tgt_feat, cross_feat
    