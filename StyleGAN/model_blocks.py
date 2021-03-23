import torch
from torch import nn
import numpy as np


class EqualizingWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        nn.init.normal_(layer.weight)
        n = np.prod(layer.weight.shape[1:])
        self.c = np.sqrt(2 / n)

    def forward(self, x):
        self.layer.weight.data = self.layer.weight.data * self.c
        x = self.layer(x)
        self.layer.weight.data = self.layer.weight.data / self.c
        return x


class AdaIN3D(nn.Module):
    def __init__(self, latent_size, out_channels):
        super().__init__()
        self.layer = EqualizingWrapper(nn.Linear(latent_size, out_channels * 2))
        self.layer.layer.bias.data.fill_(1)

    def forward(self, x, latent_vector):
        # Instance Normalization
        mean = x.mean(dim=-1).mean(dim=-1).mean(dim=-1)[..., None, None, None]
        std = x.std(dim=-1).std(dim=-1).std(dim=-1)[..., None, None, None]
        x = (x - mean) / torch.sqrt(std ** 2 + 1e-8)
        # Adaptation with style
        y = self.layer(latent_vector)
        shape = [-1, 2, x.shape[1]] + (x.dim() - 2) * [1]
        y = y.view(shape)

        x = x * y[:, 0] + y[:, 1]
        return x


class MappingNetwork(nn.Module):
    def __init__(self, latent_size, depth=8):
        super().__init__()
        blocks = [
            [EqualizingWrapper(nn.Linear(latent_size, latent_size)), nn.LeakyReLU(0.2)]
            for d in range(depth)
        ]
        # flatten the list
        layers = [l for block in blocks for l in block]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SequentialWithAdaIN(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(list(args))
        self.start_with_adain = type(self.layers[0]) == AdaIN3D

    def forward(self, x, w):
        start_idx = 0
        if self.start_with_adain:
            x = self.layers[0](x, w)
            start_idx = 1
        for i in range(start_idx, len(self.layers), 2):
            x = self.layers[i](x)
            x = self.layers[i + 1](x, w)
        return x


class NoiseAdder3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(
                x.size(0),
                1,
                x.size(2),
                x.size(3),
                x.size(4),
                device=x.device,
                dtype=x.dtype,
            )
        return x + self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)


class MiniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = x.std(dim=0).mean()
        x = torch.cat([x, torch.ones_like(x[:, 0:1]) * std], dim=1)
        return x


def ConvBlock3D(in_channels, out_channels, kernel_size=3, padding=0, relu_leak=0.2):
    return nn.Sequential(
        EqualizingWrapper(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        ),
        NoiseAdder3D(out_channels),
        nn.LeakyReLU(relu_leak),
    )


def GeneratorBlock3D(in_channels, out_channels, latent_size):
    return SequentialWithAdaIN(
        ConvBlock3D(in_channels, out_channels, padding=1),
        AdaIN3D(latent_size, out_channels),
        ConvBlock3D(out_channels, out_channels, padding=1),
        AdaIN3D(latent_size, out_channels),
    )
