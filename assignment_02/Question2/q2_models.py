"""Architectures prescribed in Question 2, without teacher forcing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from q2_core import G, apply_symmetry, inverse_symmetry


class ConvLSTMCell(nn.Module):
    def __init__(self, inputs, hidden=64, dilation=1):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(inputs + hidden, 4 * hidden, 3,
                               padding=dilation, dilation=dilation, bias=True)

    def forward(self, x, state=None):
        if state is None:
            h = x.new_zeros(x.shape[0], self.hidden, *x.shape[-2:])
            c = torch.zeros_like(h)
        else:
            h, c = state

        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Forecaster(nn.Module):
    def __init__(self, dilations=(1, 1)):
        super().__init__()
        self.layer1 = ConvLSTMCell(16, dilation=dilations[0])
        self.layer2 = ConvLSTMCell(64, dilation=dilations[1])
        self.output = nn.Conv2d(64, 16, 1, bias=True)
        # Longest one-step path: layer-1 gates then layer-2 gates, 4 pixels per cell.
        self.reach = 4 * (dilations[0] + dilations[1])
        self.reach_axis = self.reach
        self.reach_euclidean = math.sqrt(2) * self.reach_axis

    def forward(self, x):
        if tuple(x.shape[1:]) != (5, 1, 64, 64):
            raise ValueError('Expected (B,5,1,64,64) input.')

        state1 = state2 = None
        predictions = []

        for step in range(9):
            frame = x[:, step] if step < 5 else prediction
            state1 = self.layer1(F.pixel_unshuffle(frame, 4), state1)
            state2 = self.layer2(state1[0], state2)
            prediction = torch.sigmoid(F.pixel_shuffle(self.output(state2[0]), 4))
            if step >= 4:
                predictions.append(prediction)

        return torch.stack(predictions, dim=1)


class Blank(nn.Module):
    def forward(self, x):
        return x.new_zeros(x.shape[0], 5, 1, 64, 64)


class CopyLast(nn.Module):
    def forward(self, x):
        return x[:, -1:].expand(-1, 5, -1, -1, -1)


class Symmetrised(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Sequential passes keep peak memory close to one ordinary prediction.
        predictions = [inverse_symmetry(self.model(apply_symmetry(x, g)), g) for g in G]
        return torch.stack(predictions).mean(dim=0)


def build_model(name):
    if name == 'Blank':
        return Blank()

    if name == 'Copy-last':
        return CopyLast()

    if name not in ('A', 'B', 'C', 'D'):
        raise ValueError(name)

    # D: dilations 1 then 3 cover every cell offset -4..4 (no gridding), reach 16 px.
    model = Forecaster(dilations=(1, 3) if name == 'D' else (1, 1))
    assert count_parameters(model) == 480_784
    assert name != 'D' or count_parameters(model) <= 721_176
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
