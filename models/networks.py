# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from utils import persistence
from torch.nn.functional import silu
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
#Add and Group norm
class AddGroupNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(AddGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=eps)

    def forward(self, x, x_skip):
        out = x + x_skip      # Add
        out.to(x.device)
        self.norm = self.norm.to(x.device)
        out = self.norm(out)  # Norm
        return out


# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    # def forward(self, x, up=False, ref=None):
    def forward(self, x, up=False, ref_hist=None,ref_future=None):
        w = self.weight.to(x.dtype) if self.weight is not None else None

        b = self.bias.to(x.dtype) if self.bias is not None else None
        # print(f"Shape of w: {w.shape if w is not None else 'None'}")
        # print(f"Shape of b: {b.shape if b is not None else 'None'}")

        
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                # print('shape of x:', x.shape)
                # print('shape of w:', w.shape if w is not None else 'None')
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))

        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None
    ):
        super().__init__()
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    # def forward(self, x, emb, up=False, ref=None):ref_hist=None,ref_future=None
    def forward(self, x, emb, up=False, ref_hist=None,ref_future=None):
        #print(f"Shape of x: {x.shape}")
        #print(f"shape of ref: {ref.shape if ref is not None else 'None'}")
        orig = x
        #x = self.conv0(silu(self.norm0(x)))
        #print(f"Shape of x before conv0: {x.shape}")
        x = self.conv0(silu(self.norm0(x)))
        # print(f"Shape of x after conv0: {x.shape}")
        
        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = silu(self.norm1(x.add_(params)))
        else:
            x = silu(self.norm1(x))
        
        # print(f"Shape of x after + emb: {x.shape}")


        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        #print('Shape of x before attention: ',x.shape)

        if self.num_heads:
           
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x_skip = x.clone()
            C = x.shape[1]
            add_norm = AddGroupNorm(num_channels=C).to(x.device)

            
            x = self.proj(a.reshape(*x.shape))
            x = add_norm(x, x_skip)
            x = x * self.skip_scale
            
        return x



#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

## Cross Attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, top_k):
        super().__init__()
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = out_channels // num_heads
        assert out_channels % num_heads == 0

        self.q_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.k_proj = nn.Conv2d(top_k * out_channels, in_channels, kernel_size=1)
        # self.v_proj = nn.Conv2d(top_k * out_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    def forward(self, x, ref_hist, ref_future, top_k, block, block_ref_hist, block_ref_future, emb=None, block_idx='unknown', epoch=0, num_epochs=25):
        # print("x trong forward cross ")

        B, C, H, W = x.shape
        # self.head_dim = C // self.num_heads
        torch.set_printoptions(threshold=float('inf'), precision=3, linewidth=200)
        q = self.q_proj(x)  # (B, C, H, W) (32,64,16,16)

        # print(f"Đây là q {q}")
        # Tách từng ref riêng biệt
        # print(f"đây là shape {ref_hist.shape} {ref_hist} ") #(32,3,16,16)
        ref_hist_list = list(torch.chunk(ref_hist, chunks=self.top_k, dim=1)) #(32,1,16,16)
        ref_future_list = list(torch.chunk(ref_future, chunks=self.top_k, dim=1))
        
        # Xử lý từng ref_hist để tạo K_i
        k_list = []
        for r_hist in ref_hist_list:
            if isinstance(block_ref_hist, UNetBlock):
                k_i = block_ref_hist(r_hist, emb=emb) if emb is not None else block_ref_hist(r_hist)
                k_i = self.k_proj(k_i)
                # print(f"hi 111 {k_i}") 
            else:
                k_i = block_ref_hist(r_hist)
                k_i = self.k_proj(k_i)
                # print(f"hi else {k_i}")
            k_list.append(k_i)
        
        # Xử lý từng ref_future để tạo V_i  
        v_list = []
        for r_future in ref_future_list:
            if isinstance(block_ref_future, UNetBlock):
                v_i = block_ref_future(r_future, emb=emb) if emb is not None else block_ref_future(r_future)
                v_i = self.v_proj(v_i)
            else:
                v_i = block_ref_future(r_future)
                v_i = self.v_proj(v_i)
            v_list.append(v_i)
        
        def prepare(t):
            return t.reshape(B, self.num_heads, self.head_dim, H, W) \
                    .flatten(3) \
                    .permute(0, 1, 3, 2) \
                    .reshape(B * self.num_heads, H * W, self.head_dim)

        q = prepare(q) 
        
        # Tính attention với từng K_i riêng biệt
        attention_scores = []
        for k_i in k_list:
            k_i = prepare(k_i)
            score_i = torch.bmm(q, k_i.transpose(1, 2)) / (self.head_dim ** 0.5)
            attention_scores.append(score_i)  #(128,256,256) mỗi cái i
            
    
        # Softmax across all refs
        all_scores = torch.stack(attention_scores, dim=1)  # (B*heads, top_k, HW, HW)   (128,3,256,256)
        attention_weights = F.softmax(all_scores, dim=1)  # Softmax theo dimension top_k  (128,3,256,256)
        # Attn map
        # self.visualize_attention_map(attention_weights, H, W, batch_size=B, block_idx=block_idx, epoch=epoch, num_epochs=num_epochs)

        # Weighted sum với từng V_i
        output = torch.zeros_like(q) 
        for i, v_i in enumerate(v_list):
            v_i = prepare(v_i) #(128, 256, 4)
            weighted_v_i = torch.bmm(attention_weights[:, i, :, :], v_i) #(128,256,256) x (128,256,4)
            output += weighted_v_i
        
        out = output.reshape(B, self.num_heads, H * W, self.head_dim) \
                    .permute(0, 1, 3, 2) \
                    .reshape(B, C, H, W) # (32, 64, 16, 16)
        
        # Return updated refs
        updated_hist = torch.cat(k_list, dim=1)  # (32, 192, 16, 16)
        updated_future = torch.cat(v_list, dim=1) 
        # print(f"day la updated_hist {updated_hist.shape} ")
        return out, updated_hist, updated_future

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        top_k = 10,                          # Number of top-k time series to use for cross-attention.    
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.top_k = top_k
        
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        # self.enc_ref = torch.nn.ModuleDict()
        self.enc_ref_hist = torch.nn.ModuleDict()
        self.enc_ref_future = torch.nn.ModuleDict()
        self.enc_cross_attn = torch.nn.ModuleDict()
        cout = in_channels
        # print("========")
        # print(channel_mult)
        for level, mult in enumerate(channel_mult):

            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                # self.enc_ref[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                self.enc_ref_hist[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                self.enc_ref_future[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                self.enc_cross_attn[f'{res}x{res}_conv'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
            
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                # self.enc_ref[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                self.enc_ref_hist[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                self.enc_ref_future[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                self.enc_cross_attn[f'{res}x{res}_down'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
            
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                # self.enc_ref[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                self.enc_ref_hist[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                self.enc_ref_future[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                self.enc_cross_attn[f'{res}x{res}_block{idx}'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
                
        
        skips_x = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        # self.dec_ref = torch.nn.ModuleDict()
        self.dec_ref_hist = torch.nn.ModuleDict()
        self.dec_ref_future = torch.nn.ModuleDict()
        self.dec_cross_attn = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            # print level, channel_mult
            #print(f"level: {level}, res: {res}, channel_mult: {channel_mult}")
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                # self.dec_ref[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec_ref_hist[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec_ref_future[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec_cross_attn[f'{res}x{res}_in0'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
                
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                # self.dec_ref[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                self.dec_ref_hist[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                self.dec_ref_future[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                self.dec_cross_attn[f'{res}x{res}_in1'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
                
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                # self.dec_ref[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                self.dec_ref_hist[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                self.dec_ref_future[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                self.dec_cross_attn[f'{res}x{res}_up'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
                
            for idx in range(num_blocks + 1):
                cin = cout + skips_x.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                # self.dec_ref[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                self.dec_ref_hist[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                self.dec_ref_future[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)

                self.dec_cross_attn[f'{res}x{res}_block{idx}'] = CrossAttentionBlock(in_channels=cout, out_channels=cout, num_heads=4, top_k = self.top_k)
                
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

        

    def forward(self, x, ref_hist,ref_future, top_k, noise_labels, class_labels, augment_labels=None, epoch=0, num_epochs=25):
        # Mapping.
        # print(f"DhariwalUNet - Epoch: {epoch}, Num_epochs: {num_epochs}")
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        # skips_ref = [] 
        skips_ref_hist = []
        skips_ref_future = []
        # print(f"Shape of x before Unet: {x.shape}")
        # print(x)
        # x = x*mask or x*(1-mask)
        for key in self.enc.keys():
            block = self.enc[key]
            # print(f"hell {x.shape}")
        
            # block_ref = self.enc_ref[key]
            block_ref_hist = self.enc_ref_hist[key]
            block_ref_future = self.enc_ref_future[key]
            block_cross = self.enc_cross_attn[key]

            if isinstance(block, UNetBlock):
                x = block(x, emb)
            else:
                x = block(x)

            x_skip = x.clone()
            # x, ref = block_cross(x, ref, top_k=top_k, block=block, block_ref=block_ref, emb=emb, block_idx=key, epoch=epoch, num_epochs=num_epochs)
            x, ref_hist, ref_future = block_cross(x, ref_hist, ref_future, top_k=top_k, block=block, 
                                     block_ref_hist=block_ref_hist, block_ref_future=block_ref_future, 
                                     emb=emb, block_idx=key, epoch=epoch, num_epochs=num_epochs)
            # x, ref = block_cross(x, ref, top_k=top_k, block=block, emb=emb, block_idx=key, epoch=epoch, num_epochs=num_epochs)
            C = x.shape[1]
            add_norm = AddGroupNorm(num_channels=C).to(x.device)
            x = add_norm(x, x_skip)
            skips.append(x)
            # skips_ref.append(ref)
            skips_ref_hist.append(ref_hist)
            skips_ref_future.append(ref_future)
            # print(f"[DEBUG] After block {key}: x.shape = {x.shape}")
            # print(f"[DEBUG] After block {key}: ref_hist.shape = {ref_hist.shape}")
            # print(f"[DEBUG] After block {key}: ref_future.shape = {ref_future.shape}")

        # Decoder.
        for key in self.dec.keys():
            block = self.dec[key]
            # block_ref = self.dec_ref[key] 
            block_ref_hist = self.dec_ref_hist[key]
            block_ref_future = self.dec_ref_future[key]
            block_cross = self.dec_cross_attn[key]

            if x.shape[1] != block.in_channels:
                #print(f"Shape of x before attention: {x.shape}")
                x = torch.cat([x, skips.pop()], dim=1)

            x = block(x, emb)
            # x, ref_hist, ref_future = block_cross(x, ref_hist, ref_future, top_k=top_k, block=block,
            #                         block_ref_hist=block_ref_hist, block_ref_future=block_ref_future,
            #                         emb=emb, block_idx=key, epoch=epoch, num_epochs=num_epochs)
            # skips_ref_hist.append(ref_hist)
            # skips_ref_future.append(ref_future)
        #     print(f"Shape of x after attention: {x.shape}")
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        top_k =  10,                        # Number of top-k time series to use for cross-attention.   
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.top_k = top_k
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, top_k = top_k, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, ref_hist,ref_future, top_k, class_labels=None, force_fp32=False, epoch=0, num_epochs=25, **model_kwargs):
        # print(f"EDMPrecond - Epoch: {epoch}, Num_epochs: {num_epochs}")
        x = x.to(torch.float32) # his + fut noise  (32, 1, 16, 16) 
        # print(f"Shape of x networks: {x.shape}") 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), ref_hist,ref_future, top_k, c_noise.flatten(), class_labels=class_labels, epoch=epoch, num_epochs=num_epochs, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------