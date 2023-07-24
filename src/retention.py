import math

import torch
import torch.nn as nn

from xpos_relative_position import XPOS

class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        
        self.xpos = XPOS(hidden_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        att = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return att @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, hidden_size)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, hidden_size)
        # s_n_1: (batch_size, hidden_size, hidden_size)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to a slice of X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X[:, :, i*self.head_size:(i+1)*self.head_size]))
        
        Y = torch.cat(Y, dim=2)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    
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
                x_n[:, :, i*self.head_size:(i+1)*self.head_size], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=1)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(x_n.shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns
