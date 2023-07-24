import unittest
import torch
from retention import SimpleRetention, MultiScaleRetention

class TestSimpleRetention(unittest.TestCase):
    def test_simple_retention_parallel(self):
        batch_size = 4
        hidden_size = 8
        sequence_length = 16
        gamma = 0.9

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = SimpleRetention(hidden_size, gamma)

        Y = retention(X)
        self.assertEqual(Y.shape, (batch_size, sequence_length, hidden_size))
    
    def test_simple_retention_recurrent(self):
        batch_size = 4
        hidden_size = 8
        sequence_length = 16
        gamma = 0.9

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = SimpleRetention(hidden_size, gamma)

        s_n_1 = torch.zeros(hidden_size, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        Y = []
        for i in range(sequence_length):
            y_n, s_n = retention.forward_recurrent(X[:, i, :], s_n_1, i+1)
            Y.append(y_n)
            s_n_1 = s_n
        Y = torch.stack(Y, dim=1)
        self.assertEqual(Y.shape, (batch_size, sequence_length, hidden_size))
    
    def test_paradigms_identical(self):
        """
            check that the parallel and recurrent paradigms have identical outputs
        """
        batch_size = 1
        hidden_size = 8
        sequence_length = 4
        gamma = 0.90

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = SimpleRetention(hidden_size, gamma)

        Y_parallel = retention(X)

        s_n_1 = torch.zeros(hidden_size, hidden_size, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_n = retention.forward_recurrent(X[:, i, :], s_n_1, i+1)
            Y_recurrent.append(y_n)
            s_n_1 = s_n
        Y_recurrent = torch.stack(Y_recurrent, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent))

class TestMultiScaleRetention(unittest.TestCase):
    def test_multiscale_retention_parallel(self):
        batch_size = 4
        sequence_length = 5
        hidden_size = 32
        heads = 4
        retention = MultiScaleRetention(hidden_size, heads)

        X = torch.rand(batch_size, sequence_length, hidden_size)
        Y = retention(X)
        self.assertEqual(Y.shape, (batch_size, sequence_length, hidden_size))

    def test_multiscale_retention_recurrent(self):
        batch_size = 4
        sequence_length = 5
        hidden_size = 32
        heads = 4
        retention = MultiScaleRetention(hidden_size, heads)

        X = torch.rand(batch_size, sequence_length, hidden_size)
        s_n_1s = [
            torch.zeros(hidden_size // heads, hidden_size // heads, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        Y = []
        for i in range(sequence_length):
            y_n, s_ns = retention.forward_recurrent(X[:, i, :], s_n_1s, i)
            Y.append(y_n)
            s_n_1s = s_ns
        Y = torch.stack(Y, dim=1)
        self.assertEqual(Y.shape, (batch_size, sequence_length, hidden_size))
    
    def test_multiscale_paradigms_identical(self):
        """
            check that the parallel and recurrent paradigms have identical outputs
        """
        batch_size = 2
        hidden_size = 36
        sequence_length = 5
        heads = 3

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = MultiScaleRetention(hidden_size, heads)

        Y_parallel = retention(X)

        s_n_1s = [
            torch.zeros(hidden_size // heads, hidden_size // heads, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retention.forward_recurrent(X[:, i, :], s_n_1s, i)
            Y_recurrent.append(y_n)
            s_n_1s = s_ns
        Y_recurrent = torch.stack(Y_recurrent, dim=1)

        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent))

unittest.main()