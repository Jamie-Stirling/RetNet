import unittest
import torch
from retnet import RetNet, RetNetCLM

class TestRetNet(unittest.TestCase):
    
    def test_paradigms_equivalent(self):
        batch_size = 2
        layers = 2
        hidden_dim = 8
        heads = 4
        sequence_length = 4
        ffn_size = 16

        X = torch.rand(batch_size, sequence_length, hidden_dim)

        retnet = RetNet(layers, hidden_dim, ffn_size, heads)
        Y_parallel = retnet(X)

        s_n_1s = [
            [
                torch.zeros(hidden_dim // heads, hidden_dim // heads, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
                for _ in range(heads)
            ] for _ in range(layers)
        ]

        Y_recurrent = []
        for i in range(sequence_length):
            Y, s_ns = retnet.forward_recurrent(X[:, i, :], s_n_1s, i+1)
            Y_recurrent.append(Y)
            s_n_1s = s_ns
        
        Y_recurrent = torch.stack(Y_recurrent, dim=1)

        print((Y_parallel - Y_recurrent).abs().max())

        self.assertTrue((Y_parallel - Y_recurrent).abs().max() < 1e-4)

    def test_clm(self):
        batch_size = 2
        layers = 2
        hidden_dim = 16
        heads = 4
        sequence_length = 6
        ffn_size = 32
        vocab_size = 10

        X = torch.randint(0, vocab_size, (batch_size, sequence_length))

        retnet = RetNetCLM(layers, hidden_dim, ffn_size, heads, vocab_size)
        Y_parallel = retnet(X)

        s_n_1s = [
            [
                torch.zeros(hidden_dim // heads, hidden_dim // heads, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
                for _ in range(heads)
            ] for _ in range(layers)
        ]

        Y_recurrent = []
        for i in range(sequence_length):
            Y, s_ns = retnet.forward_recurrent(X[:, i], s_n_1s, i+1)
            Y_recurrent.append(Y)
            s_n_1s = s_ns
        
        Y_recurrent = torch.stack(Y_recurrent, dim=1)

        # test sample
        Y_sample = retnet.sample(X, 5)

        self.assertTrue(Y_sample.shape == (batch_size, 5))
        
        self.assertTrue((Y_parallel - Y_recurrent).abs().max() < 1e-4)
    
    def test_training(self):
        batch_size = 2
        layers = 3
        hidden_dim = 16
        heads = 4
        sequence_length = 6
        ffn_size = 32
        vocab_size = 10
        bos_idx = 0

        data = torch.randint(0, vocab_size, (batch_size, sequence_length - 1))
        X = torch.cat([torch.ones(batch_size, 1).long() * bos_idx, data[:,:-1]], dim=1)
        Y = data

        # verify we can overfit autoregressive model
        model = RetNetCLM(layers, hidden_dim, ffn_size, heads, vocab_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(model(X).reshape(-1, 10), Y.reshape(-1))
        for i in range(10):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.reshape(-1, 10), Y.reshape(-1))
            loss.backward()
            optimizer.step()
        self.assertTrue((loss < initial_loss).item())
unittest.main()