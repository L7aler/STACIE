import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding as used in the traditional transformer. Adds a 'positional encoding' vector to the feature
    vector, which allows the model to derive the position of a datapoint from the features.
    d_model: number of features in the data
    max_len: maximum size of an source sequence
    dropout: dropout probability
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, src):
        # src has shape [batch_size, sequence_length, d_model]
        # self.pe has shape [1, max_len, d_model]
        src = src + self.pe[:, :src.size(1)]
        return self.dropout(src)


def t2v(tau, f, w, b, w0, b0, arg=None):
    """
    Time2Vec function. From timestep scalar tau, a vector that represents that timestep is created.
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    """
    A Time2Vec module with a sine as its periodic function (self.f). This is an alternative to positional encoding,
    which should work better for time series (instead of e.g. natural language problems).
    """

    def __init__(self, out_features, device):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin
        self.device = device

    def forward(self, x):
        t = torch.FloatTensor(torch.arange(x.size(1), dtype=torch.float32)).unsqueeze(-1).to(self.device)
        v = t2v(t, self.f, self.w, self.b, self.w0, self.b0).repeat((x.size(0), 1, 1))
        return torch.cat((x, v), dim=-1)


class TimeDelay(nn.Module):
    """
    Time Delay positional embedding. For each time step, concatenates the 'dim' previous steps with 'delay'
    step interval. There is no padding, so you lose some timesteps from the start of sequence based on the 'dim' and
    'delay'.
    Example:
        data = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
        dim = 3
        delay = 2
        new_data = [[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9]]
    """

    def __init__(self, dim, delay):
        super(TimeDelay, self).__init__()
        self.dim = dim
        self.delay = delay

    def embedding(self, x, i):
        # Concatenate step i with some previous steps (x is reversed)
        steps = x[:, i:i + self.dim * self.delay:self.delay]  # [batch, dim, features]
        steps = steps.flip(dims=[1])
        steps = steps.view(x.size(0), 1, -1)  # [batch, 1, dim * features]
        return steps

    def forward(self, x):
        new_sequence_length = x.size(1) - self.dim * self.delay
        # Reverse the order to start with the last step
        x = x.flip(dims=[1])
        # For each step, collect 'dim' previous data points spaced by 'delay' steps
        # Also reverse the order of each collection
        embeddings = [self.embedding(x, i) for i in range(new_sequence_length)]
        # Concatenate the new data
        x = torch.cat(embeddings, dim=1)  # [batch, new_sequence_length, dim * features]
        # Reverse the order back
        x = x.flip(dims=[1])
        return x
