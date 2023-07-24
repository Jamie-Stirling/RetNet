import math
import torch
import torch.nn as nn

class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(ComplexGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        X is assumed to be complex
        """
        X = X.reshape(-1, self.num_groups, self.num_channels // self.num_groups)
        mean = X.mean(dim=2, keepdim=True)
        var = X.var(dim=2, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X.reshape(-1, self.num_channels)
        X = X * self.weight + self.bias
        
        return X
    
class ComplexLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))

    def forward(self, X):
        """
        X: unknown shape ending in hidden_size
        we treat the last dimension as the hidden_size
        """
        X_shape = X.shape
        X = X.reshape(-1, X_shape[-1])
        mean = X.mean(dim=1, keepdim=True)
        var = X.abs().var(dim=1, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X * self.weight + self.bias
        X = X.reshape(X_shape)
        return X


class ComplexFFN(nn.Module):
    """
    2 linear layers with no bias
    """
    def __init__(self, hidden_size, ffn_size):
        super(ComplexFFN, self).__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_size, ffn_size, dtype=torch.float32) / math.sqrt(hidden_size))
        self.W2 = nn.Parameter(torch.randn(ffn_size, hidden_size, dtype=torch.float32) / math.sqrt(ffn_size))
        self.gelu = lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        X is assumed to be complex
        """
        # reshaping
        X = X @ self.W1.to(X)
        X = self.gelu(X)
        X = X @ self.W2.to(X)
        
        return X
