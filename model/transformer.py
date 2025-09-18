import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class RMSNorm(torch.nn.Module):

    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):

    def __init__(self, config, embed_dim, n_head):
        super().__init__()
        assert embed_dim % n_head == 0
        # query
        if config.n_query >= 1:
            self.q_token = nn.Parameter(torch.randn(1, config.n_query, embed_dim) * 0.02)
        # key, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 2 * embed_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = n_head
        self.n_embd = embed_dim
        self.dropout = config.dropout
        self.n_query = config.n_query
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, n_merge=-1, time=-1, query=None, mask=None):
        origin_B = x.size(0)
        if time == -1:
            x = rearrange(x, 'B (A N) C -> (B A) N C', N=n_merge)
        else:
            x = rearrange(x, 'B (N T) C -> (B T) N C', T=time)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if query is None:
            q_token = self.q_token.expand(B, -1, -1)
            q = q_token.reshape(
                B, self.n_query, self.n_head, C // self.n_head
            ).transpose(1, 2)
        else:
            q = query.reshape(
                B, query.size(1), self.n_head, C // self.n_head
            ).transpose(1, 2)
        k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, self.n_query, C) # re-assemble all head outputs side by side
        if time == -1:
            y = rearrange(y, '(B A) N C -> B (A N) C', B=origin_B)
        else:
            y = rearrange(y, '(B T) N C -> B (N T) C', B=origin_B)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rmsn_1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.rmsn_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.rmsn_1(x), mask)
        x = x + self.mlp(self.rmsn_2(x))
        return x
