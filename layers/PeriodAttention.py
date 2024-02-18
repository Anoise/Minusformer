#!/usr/bin/env python
"""
@author : Daojun Liang
@email  : daojunliang@gmail.com
@time   : 2022/7/27 15:38
@desc   : PeriodAttention.py
"""

from operator import mod
import torch
import torch.nn as nn

class PeriodAttention(nn.Module):
    def __init__(self, period=None, scale=0.45, attention_dropout=0., output_attention=False, padding=True):
        super(PeriodAttention, self).__init__()
        self.period = period
        self.output_attention = output_attention
        self.attn_drop = nn.Dropout(attention_dropout)
        self.padding = padding
        self.scale = scale
        
    def forward(self, q, k, v, attn_mask=None):
        #B, L, D = q.shape
        B, L, E = q.shape
        _, S, D = v.shape

        if L > S:
            #zeros = torch.zeros_like(q).float()
            zeros = torch.zeros(B,L-S,D).float().to(v.device)
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        elif L < S:
            v = v[:, :L, :]
            k = k[:, :L, :]
        
        # obtain input's period
        if self.period is None:
            period = L//4
        else:
            period = self.period
        
        if L%period !=0:
            if self.padding:
                zeros = torch.zeros(B,period-L%period,D).float().to(v.device)
                q =torch.cat([q, zeros], dim=1)
                v =torch.cat([v, zeros], dim=1)
                k = torch.cat([k, zeros], dim=1)
            else:
                q = q[:, :-L%period, :]
                v = v[:, :-L%period, :]
                k = k[:, :-L%period, :]
                out_pad = v = v[:, -L%period:, :]
        
        q, k, v = q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1)
        _L = q.size(-1)
        
        q = q.view(B, D, _L // period, period)
        k = k.view(B, D, _L // period, period)
        v = v.view(B, D, _L // period, period)
        if self.scale != 0: 
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        else:
            attn = None
            out = self.attn_drop(self.act(v))
        out = out.view(B,D,-1)
        out = out.permute(0, 2, 1)
        if self.padding:
            out = out[:,:L,:] 
        else:
            out =torch.cat([out, out_pad], dim=1)
        return out, attn



class PeriodAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(PeriodAttentionLayer, self).__init__()

        d_qkv = d_model // n_heads

        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_qkv)
        self.key_projection = nn.Linear(d_model, d_qkv)
        self.value_projection = nn.Linear(d_model, d_qkv)
        self.out_projection = nn.Linear(d_qkv, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        out, attn = self.attention(
            queries,
            keys,
            values,
            attn_mask
        )
        return self.out_projection(out), attn
