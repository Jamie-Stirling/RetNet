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

    def test_simple_retention_chunkwise(self):
        batch_size = 2
        hidden_size = 5
        sequence_length = 16
        chunk_size = 2
        gamma = 0.9

        X = torch.rand(batch_size, sequence_length, hidden_size)
        retention = SimpleRetention(hidden_size, gamma)

        r_i_1 = torch.zeros(hidden_size, hidden_size, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)

        Y = []
        for i in range(sequence_length // chunk_size):
            x_i = X[:, i * chunk_size : (i + 1) * chunk_size, :]
            y_i, r_i = retention.forward_chunkwise(x_i, r_i_1, i, chunk_size)
            Y.append(y_i)
            r_i_1 = r_i
        
        Y = torch.cat(Y, dim=1)
        self.assertEqual(Y.shape, (batch_size, sequence_length, hidden_size))

    
    def test_paradigms_identical(self):
        """
            check that the parallel and recurrent paradigms have identical outputs
        """
        batch_size = 2
        hidden_size = 5
        sequence_length = 32
        chunk_size = 2
        gamma = 0.999

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

        r_n_1 = torch.zeros(hidden_size, hidden_size, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
        Y_chunkwise = []

        for i in range(1, sequence_length // chunk_size + 1):
            x_i = X[:, (i-1) * chunk_size : (i) * chunk_size, :]
            y_i, r_n = retention.forward_chunkwise(x_i, r_n_1, i, chunk_size)
            Y_chunkwise.append(y_i)
            r_n_1 = r_n
        
        Y_chunkwise = torch.cat(Y_chunkwise, dim=1)

        print(Y_parallel[0, 1, 2].item())
        print(Y_recurrent[0, 1, 2].item())
        print(Y_chunkwise[0, 1, 2].item())

        print((Y_parallel - Y_recurrent).abs().max())
        print((Y_parallel - Y_chunkwise).abs().max())
        
        self.assertTrue(torch.allclose(Y_parallel, Y_recurrent))
        self.assertTrue(torch.allclose(Y_parallel, Y_chunkwise))

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
