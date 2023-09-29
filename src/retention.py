import math
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, einsum

from xpos_relative_position import XPOS


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, W_Q=None, W_K=None, W_V=None, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        if W_Q is None:
            self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        else:
            self.W_Q = W_Q
        if W_K is None:
            self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        else:
            self.W_K = W_K
        if W_V is None:
            self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        else:
            self.W_V = W_V

        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        # e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D


def _build_decay_gammas(
        num_heads: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    xmin, xmax = math.log(1 / 32), math.log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - torch.exp(x)


def _build_decay_mask(
        query_length: int,
        key_length: int,
        decay_gammas: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)

    distance = torch.abs(query_pos.unsqueeze(-1) - key_pos.unsqueeze(0))
    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool, device=device).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas ** distance


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = self.head_size * 2 if double_v_dim else self.head_size
        self.xpos = XPOS(self.head_size)

        self.swish = nn.SiLU()

        self.W_Q = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_v_dim) / hidden_size)

        self.gammas_list = _build_decay_gammas(self.heads, self.W_Q.device, self.W_Q.dtype)
        self.gammas = nn.Parameter(self.gammas_list,
                                   requires_grad=False)
        self.gammas_list = deepcopy(self.gammas_list).detach().cpu().tolist()

        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.W_Q[i], self.W_K[i],
                            self.W_V[i], self.head_size, double_v_dim)
            for i, gamma in enumerate(self.gammas_list)
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        b n l dim
        """
        # b=batch l=sq_len d=hiddendim n=n-head h=headdim v=vdim
        batch, sq_len, _ = X.shape
        _, key_len, _ = X.shape
        q_proj = einsum(X, self.W_Q, "b l d, n d h -> b n l h")
        k_proj = einsum(X, self.W_K, "b l d, n d h -> b n l h")
        v_proj = einsum(X, self.W_V, "b l d, n d v -> b n l v")

        q_proj = rearrange(q_proj, "b n l h -> (b n) l h")
        k_proj = rearrange(k_proj, "b n l h -> (b n) l h")
        q_proj = self.xpos(q_proj)
        k_proj = self.xpos(k_proj, downscale=True)
        decay = _build_decay_mask(sq_len, key_len, self.gammas, q_proj.device, dtype=q_proj.dtype).unsqueeze(0)
        ret = q_proj @ k_proj.permute(0, 2, 1)
        ret = rearrange(ret, "(b n) l h -> b n l h", b=batch)
        ret = (ret * decay) @ v_proj
        ret = rearrange(ret, "b n l h -> b l (n h)")
        # apply each individual retention mechanism to X
        ret_shape = ret.shape
        ret = self.group_norm(ret.reshape(-1, self.v_dim)).reshape(ret_shape)

        return (self.swish(X @ self.W_G) * ret) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """

        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
            )
            Y.append(y)
            s_ns.append(s_n)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
            )
            Y.append(y)
            r_is.append(r_i)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is
