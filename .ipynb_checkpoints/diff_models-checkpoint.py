import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm
from einops import repeat, rearrange
from torch import einsum

def default(val, d):
    return val if val is not None else d

class ReferenceModulatedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.y_to_q = nn.Linear(dim, inner_dim, bias = False)
        self.cond_to_k = nn.Linear(2*dim+context_dim, inner_dim, bias = False)
        self.ref_to_v = nn.Linear(dim+context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        cond_info,
        reference,
        return_attn = False,
    ):
        B, C, K, L, h, device = x.shape[0], x.shape[1], x.shape[2], x.shape[-1], self.heads, x.device
        x = self.norm(x)
        reference = self.norm(reference)
        cond_info = self.context_norm(cond_info)
        reference = repeat(reference, 'b n c -> (b f) n c', f=C)# (B*C, K, L)
        q_y = self.y_to_q(x.reshape(B*C, K, L))# (B*C,K,ND)
        
        cond=self.cond_to_k(torch.cat((x.reshape(B*C, K, L), cond_info.reshape(B*C, K, L), reference), dim=-1))# (B*C,K,ND)
        ref=self.ref_to_v(torch.cat((x.reshape(B*C, K, L), reference), dim=-1))# (B*C,K,ND)
        q_y, cond, ref = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_y, cond, ref))# (B*C, N, K, D)
        sim = einsum('b h i d, b h j d -> b h i j', cond, ref) * self.scale # (B*C, N, K, K)
        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)
        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, ref)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, cond)
        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        if return_attn:
            return out, context_out, attn, context_attn

        return out
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)
    
def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Reference_Modulated_Attention(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_RATD(nn.Module):
    def __init__(self, config, inputdim=2, use_ref=True):
        super().__init__()
        self.channels = config["channels"]
        self.use_ref=use_ref
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    ref_size=config["ref_size"],
                    h_size=config["h_size"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, reference=None):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, reference)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, ref_size, h_size, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        dim_heads=8
        self.fusion_type=1
        self.q_dim=nheads*dim_heads
        self.attn1 = CrossAttention(
                query_dim=nheads*dim_heads,
                heads=nheads,
                dim_head=dim_heads,
                dropout=0,
                bias=False,
            )
        self.RMA=ReferenceModulatedCrossAttention(dim=ref_size+h_size,context_dim=ref_size*3)
        self.line= nn.Linear(
                ref_size*3, ref_size+h_size
            )
        #self.line3 = nn.Linear(nheads*dim_heads, 2)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y
    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward(self, x, cond_info, diffusion_emb, reference):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        #reference = repeat(reference, 'b n c -> (b f) n c', f=inputdim)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        if reference!=None and self.fusion_type==1:
            cond_info = self.RMA(y.reshape(B, channel, K, L),cond_info.reshape(B, channel, K, L),reference)
            #reference = self.line(reference)
            #reference = torch.sigmoid(reference)# (B,K,L)
            #reference=reference.reshape(B, 1, K, L).permute(0,1,3,2)
            #reference = repeat(reference, 'b a n c -> (b a f) n c', f=2*channel)# (B*2*channel, L,K)
            #cond_info = torch.bmm(cond_info.reshape(B*2*channel, K , L), reference)# (B*2*channel, K, K)
            #cond_info = torch.sigmoid(cond_info)
            #cond_info = torch.bmm(cond_info, y.reshape(B*2*channel,K, L)).reshape(B,2*channel,K*L)
            #y = y + cond_info
        elif reference!=None and self.fusion_type==2:
            reference = self.line(reference)
            reference = torch.sigmoid(reference)# (B,K,L)
            reference = reference.reshape(B, 1, K, L)
            reference = repeat(reference, 'b a n c -> b (a f) n c', f=channel)# (B*2*channel, L,K)
            cond_info = cond_info + reference.reshape(B, channel, K*L)
            
        y = y + cond_info.reshape(B, channel, K*L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        #y = y + cond_info.reshape(B, 2*channel, K*L)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
