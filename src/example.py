import torch
import torch.nn as nn

import retnet

if __name__ == "__main__":
    # verify model size for hyperparameters in paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.3B model
    layers = 24
    hidden_dim = 2048
    ffn_size = 4096
    heads = 16

    retnet = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True).to(device)
    print("1.3B model:",sum(p.numel() for p in retnet.parameters() if p.requires_grad))
