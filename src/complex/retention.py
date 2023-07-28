import math

import torch
import torch.nn as nn

from util import ComplexGroupNorm

class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, precision="single"):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.precision = precision
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        

        self.theta = torch.randn(hidden_size) / hidden_size
        self.theta = nn.Parameter(self.theta)

        

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(X.device)

        if X.dtype != self.complex_type:
            X = torch.complex(X, torch.zeros_like(X)).to(self.complex_type)
        
        i = self.i.to(X.device)
        ns = torch.arange(1, sequence_length + 1, dtype=self.real_type, device=X.device)
        ns = torch.complex(ns, torch.zeros_like(ns)).to(self.complex_type)
        Theta = []

        for n in ns:
            Theta.append(torch.exp(i * n * self.theta))
        
        Theta = torch.stack(Theta, dim=0)
        
        Theta_bar = Theta.conj()

        Q = (X @ self.W_Q.to(self.complex_type)) * Theta.unsqueeze(0)
        K = (X @ self.W_K.to(self.complex_type)) * Theta_bar.unsqueeze(0)
        V = X @ self.W_V.to(self.complex_type)
        att = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return att @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, hidden_size)
        s_n_1: (batch_size, hidden_size)
        """
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, torch.zeros_like(x_n)).to(self.complex_type)
        
        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        Theta = torch.exp(self.i * n * self.theta)
        Theta_bar = Theta.conj()

        Q = (x_n @ self.W_Q.to(self.complex_type)) * Theta
        K = (x_n @ self.W_K.to(self.complex_type)) * Theta_bar
        V = x_n @ self.W_V.to(self.complex_type)

        # K: (batch_size, hidden_size)
        # V: (batch_size, hidden_size)
        # s_n_1: (batch_size, hidden_size, hidden_size)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + K.unsqueeze(2) @ V.unsqueeze(1)
        
        return (Q.unsqueeze(1) @ s_n).squeeze(1), s_n
    
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)
        
        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0
        
        return D

class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, precision="single"):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.precision = precision
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads, dtype=self.real_type))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.group_norm = ComplexGroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """
        if X.dtype != self.complex_type:
            X = torch.complex(X, torch.zeros_like(X)).to(self.complex_type)
        
        # apply each individual retention mechanism to a slice of X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X[:, :, i*self.head_size:(i+1)*self.head_size]))
        
        Y = torch.cat(Y, dim=2)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (self.swish(X @ self.W_G.to(self.complex_type)) * Y) @ self.W_O.to(self.complex_type)
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        """
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, torch.zeros_like(x_n)).to(self.complex_type)
        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, i*self.head_size:(i+1)*self.head_size], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=1)
        Y = self.group_norm(Y)
        return (self.swish(x_n @ self.W_G.to(self.complex_type)) * Y) @ self.W_O.to(self.complex_type), s_ns
