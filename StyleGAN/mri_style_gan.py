import torch
from torch import nn
import numpy as np
from model_blocks import *


class StyleGenerator(nn.Module):
    def __init__(
        self, hidden_channels, latent_size, max_resolution=32, start_resolution=4
    ):
        super().__init__()
        self.alpha = 1.0
        self.cur_last = int(np.log2(start_resolution)) - 2
        self.toRGB = EqualizingWrapper(nn.Conv3d(hidden_channels, 1, kernel_size=1))
        self.const_input = nn.Parameter(torch.ones(1, hidden_channels, 4, 4, 4))
        self.mapping_net = MappingNetwork(latent_size)
        self.layers = nn.ModuleList(
            [
                SequentialWithAdaIN(
                    AdaIN3D(latent_size, hidden_channels),
                    ConvBlock3D(hidden_channels, hidden_channels, 3, padding=1),
                    AdaIN3D(latent_size, hidden_channels),
                )
            ]
            + [
                GeneratorBlock3D(hidden_channels, hidden_channels, latent_size)
                for size in range(int(np.log2(max_resolution) - 1))
            ]
        )

    def forward(self, z):
        z = nn.functional.normalize(z)
        w = self.mapping_net(z)

        out = self.layers[0](self.const_input, w)
        for i in range(self.cur_last):
            up = interpolate(
                out,
                scale_factor=2,
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            out = self.layers[i + 1](up, w)
        if self.cur_last > 0:
            return self.alpha * self.toRGB(out) + (1 - self.alpha) * self.toRGB(up)
        return self.toRGB(out)

    def increase_resolution(self):
        self.cur_last += 1
        self.alpha = 0

    def increase_alpha(self):
        self.alpha = min(1, self.alpha + 0.1)


class ProgressiveDiscriminator(nn.Module):
    def __init__(self, hidden_channels, max_resolution=32, start_resolution=4):
        super().__init__()
        self.alpha = 1.0
        self.cur_last = int(np.log2(start_resolution)) - 2
        self.fromRGB = EqualizingWrapper(nn.Conv3d(1, hidden_channels, kernel_size=1))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    MiniBatchStdDev(),
                    ConvBlock3D(hidden_channels + 1, hidden_channels, 3, padding=1),
                    ConvBlock3D(hidden_channels, hidden_channels, 4),
                    nn.Flatten(),
                    EqualizingWrapper(nn.Linear(hidden_channels, 1)),
                )
            ]
            + [
                nn.Sequential(
                    ConvBlock3D(hidden_channels, hidden_channels, 3, padding=1),
                    ConvBlock3D(hidden_channels, hidden_channels, 3, padding=1),
                )
                for size in range(int(np.log2(max_resolution) - 1))
            ]
        )

    def forward(self, img):
        out = self.fromRGB(img)
        out = self.layers[self.cur_last](out)
        last_applied = self.cur_last

        if self.cur_last > 0:
            down = interpolate(
                img,
                scale_factor=0.5,
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            down = self.fromRGB(down)
            out = interpolate(
                out,
                scale_factor=0.5,
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            out = self.alpha * out + (1 - self.alpha) * down
            out = self.layers[self.cur_last - 1](out)
            last_applied = self.cur_last - 1

        for i in range(last_applied - 1, -1, -1):
            out = interpolate(
                out,
                scale_factor=0.5,
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            out = self.layers[i](out)
        return out

    def increase_resolution(self):
        self.cur_last += 1
        self.alpha = 0

    def increase_alpha(self):
        self.alpha = min(1, self.alpha + 0.1)
