import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, n_input_dim, encoding_config: dict):
        super().__init__()

        self.n_input_dim = n_input_dim