import torch
import torch.nn as nn

from retention import MultiScaleRetention
from util import ComplexFFN, ComplexGroupNorm, ComplexLayerNorm

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            ComplexFFN(hidden_dim, ffn_size)
            for _ in range(layers)
        ])
        self.layer_norm = ComplexLayerNorm(hidden_dim)
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norm(X)) + X
            X = self.ffns[i](self.layer_norm(Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norm(x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norm(y_n)) + y_n
        
        return x_n, s_ns

class RetNetCLM(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, vocab_size):
        """
        NOTE: softmax not included!
        """
        super(RetNetCLM, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.vocab_size = vocab_size

        self.retnet = RetNet(layers, hidden_dim, ffn_size, heads)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Parameter(torch.randn(hidden_dim, vocab_size, dtype=torch.float32) / hidden_dim)
    
    def forward(self, input_ids):
        """
        input_ids: (batch_size, sequence_length)
        """
        X = self.embed(input_ids)
        X = self.retnet(X)
        X = X @ self.proj.to(X.dtype)

        return X.real
    
    def forward_recurrent(self, input_ids, s_n_1s, n):
        """
        input_ids: (batch_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        X = self.embed(input_ids)
        X, s_ns = self.retnet.forward_recurrent(X, s_n_1s, n)
        X = X @ self.proj.to(X.dtype)

        return X.real, s_ns
    
    def sample(self, input_ids, sample_length, temperature=1.0):
        """
        input_ids: (batch_size, sequence_length)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        s_n_1s = [
            [
                torch.zeros(self.hidden_dim // self.heads, self.hidden_dim // self.heads, dtype=torch.complex64).unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
                for _ in range(self.heads)
            ] for _ in range(self.layers)
        ]
        for i in range(input_ids.shape[1]):
            X, s_n_1s = self.forward_recurrent(input_ids[:, i], s_n_1s, i+1)
        
        # get softmax of x (real part only)
        X = X.real / temperature
        X = torch.softmax(X, dim=-1)
        X = torch.multinomial(X, num_samples=1)
        next_char = X[:, -1]
        output_ids = []
        # now start sampling!
        for i in range(sample_length):
            X, s_n_1s = self.forward_recurrent(next_char, s_n_1s, i+1)
            X = X.real / temperature
            X = torch.softmax(X, dim=-1)
            X = torch.multinomial(X, num_samples=1)
            next_char = X[:, -1]
            output_ids.append(next_char)

        output_ids = torch.stack(output_ids, dim=1)

        return output_ids