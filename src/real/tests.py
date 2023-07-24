import unittest

import torch

from retention import SimpleRetention, MultiScaleRetention
from retnet import RetNet

class TestRetention(unittest.TestCase):

    def test_simple(self):
        """
        verify that the parallel and recurrent implementations of SimpleRetention are identical
        """
        batch_size = 4
        sequence_length = 12
        hidden_size = 32

        gamma = 0.9

        X = torch.rand(batch_size, sequence_length, hidden_size)
        sr = SimpleRetention(hidden_size, gamma)

        Y_parallel = sr(X)

        s_n_1 = torch.zeros(hidden_size).unsqueeze(0).repeat(batch_size, 1, 1)
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_n = sr.forward_recurrent(X[:, i:i+1, :], s_n_1, i)
            Y_recurrent.append(y_n)
            s_n_1 = s_n

        Y_recurrent = torch.concat(Y_recurrent, dim=1)

        assert torch.allclose(Y_parallel, Y_recurrent, atol=1e-5)
  
    def test_multiscale(self):
        """
        verify that the parallel and recurrent implementations of MultiScaleRetention are identical
        """
        batch_size = 2
        hidden_size = 36
        sequence_length = 5
        heads = 3

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = MultiScaleRetention(hidden_size, heads)

        Y_parallel = retention(X)

        s_n_1s = [
            torch.zeros(hidden_size // heads, hidden_size // heads).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retention.forward_recurrent(X[:, i:i+1, :], s_n_1s, i)
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        Y_recurrent = torch.concat(Y_recurrent, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent, atol=1e-5))

class TestRetNet(unittest.TestCase):

    def test_retnet(self):
        """
        verify that the parallel and recurrent implementations of RetNet are identical
        """
        batch_size = 2
        hidden_size = 36
        sequence_length = 5
        heads = 3
        layers = 4
        ffn_size = 128

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retnet = RetNet(layers, hidden_size, ffn_size, heads)

        Y_parallel = retnet(X)

        s_n_1s = [
            [
                torch.zeros(hidden_size // heads, hidden_size // heads).unsqueeze(0).repeat(batch_size, 1, 1)
                for _ in range(heads)
            ]
            for _ in range(layers)
        ]
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retnet.forward_recurrent(X[:, i:i+1, :], s_n_1s, i)
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        Y_recurrent = torch.concat(Y_recurrent, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
