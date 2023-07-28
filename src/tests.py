import unittest

import torch

from retention import SimpleRetention, MultiScaleRetention
from retnet import RetNet

class TestRetention(unittest.TestCase):

    def test_simple(self):
        """
        verify that the three implementations of SimpleRetention are identical
        """
        batch_size = 4
        sequence_length = 12
        hidden_size = 6
        chunk_size = 4

        gamma = 0.9

        X = torch.rand(batch_size, sequence_length, hidden_size)
        sr = SimpleRetention(hidden_size, gamma, double_v_dim=True)

        Y_parallel = sr(X)

        s_n_1 = torch.zeros(hidden_size, sr.v_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_n = sr.forward_recurrent(X[:, i:i+1, :], s_n_1, i)
            Y_recurrent.append(y_n)
            s_n_1 = s_n

        Y_recurrent = torch.concat(Y_recurrent, dim=1)

        r_n_1 = torch.zeros(hidden_size, sr.v_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        Y_chunkwise = []
        for i in range(sequence_length // chunk_size):
            y_i, r_i = sr.forward_chunkwise(X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1, i)
            Y_chunkwise.append(y_i)
            r_n_1 = r_i
            
        
        Y_chunkwise = torch.concat(Y_chunkwise, dim=1)


        assert torch.allclose(Y_parallel, Y_recurrent, atol=1e-5)
        assert torch.allclose(Y_parallel, Y_chunkwise, atol=1e-5)
      
  
    def test_multiscale(self):
        """
        verify that the three implementations of MultiScaleRetention are identical
        """
        batch_size = 2
        hidden_size = 6
        sequence_length = 12
        heads = 3
        chunk_size = 2

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = MultiScaleRetention(hidden_size, heads, double_v_dim=False)
        # print total number of parameters
        print("Default v_dim:",sum(p.numel() for p in retention.parameters() if p.requires_grad))
        
        retention = MultiScaleRetention(hidden_size, heads, double_v_dim=True)
        print("Double v_dim:",sum(p.numel() for p in retention.parameters() if p.requires_grad))

        Y_parallel = retention(X)

        s_n_1s = [
            torch.zeros(hidden_size // heads, retention.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retention.forward_recurrent(X[:, i:i+1, :], s_n_1s, i)
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        Y_recurrent = torch.concat(Y_recurrent, dim=1)

        r_n_1s = [
            torch.zeros(hidden_size // heads, retention.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        Y_chunkwise = []
        for i in range(sequence_length // chunk_size):
            y_i, r_i = retention.forward_chunkwise(X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1s, i)
            Y_chunkwise.append(y_i)
            r_n_1s = r_i

        Y_chunkwise = torch.concat(Y_chunkwise, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent, atol=1e-5))
        self.assertTrue(torch.allclose(Y_parallel, Y_chunkwise, atol=1e-5)) # fails

class TestRetNet(unittest.TestCase):

    def test_retnet(self):
        """
        verify that the three implementations of RetNet are identical
        """
        batch_size = 2
        hidden_size = 36
        sequence_length = 5
        heads = 3
        layers = 4
        ffn_size = 128

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retnet = RetNet(layers, hidden_size, ffn_size, heads, double_v_dim=False)
        # print total number of parameters
        print("Default v_dim:",sum(p.numel() for p in retnet.parameters() if p.requires_grad))

        retnet = RetNet(layers, hidden_size, ffn_size, heads, double_v_dim=True)
        print("Double v_dim:",sum(p.numel() for p in retnet.parameters() if p.requires_grad))

        Y_parallel = retnet(X)

        s_n_1s = [
            [
                torch.zeros(hidden_size // heads, retnet.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
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

        r_n_1s = [
            [
                torch.zeros(hidden_size // heads, retnet.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
                for _ in range(heads)
            ]
            for _ in range(layers)
        ]
        Y_chunkwise = []
        for i in range(sequence_length):
            y_i, r_i = retnet.forward_chunkwise(X[:, i:i+1, :], r_n_1s, i)
            Y_chunkwise.append(y_i)
            r_n_1s = r_i
        
        Y_chunkwise = torch.concat(Y_chunkwise, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent, atol=1e-5))
        self.assertTrue(torch.allclose(Y_parallel, Y_chunkwise, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
