import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
              nn.Conv3d(1, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm3d(hidden_dim),
              nn.MaxPool3d(2), #29, 35, 29
              nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm3d(hidden_dim),
              nn.MaxPool3d(2), #14, 17, 14
              nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm3d(hidden_dim),
              nn.MaxPool3d(2), #7, 8, 7
              nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm3d(hidden_dim),
              nn.MaxPool3d(2), #3, 4, 3
              nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.BatchNorm3d(hidden_dim),
              nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2),
              nn.AdaptiveAvgPool3d((1, 1, 1)),
              nn.Flatten(),
              nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (hidden_dim, 1, 1, 1)), # 1x1x1 (поменяли 1 на 64 ())
            nn.Upsample(size=(3, 4, 3), mode='trilinear',), #3x4x3
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(hidden_dim),
            nn.Upsample(size=(7, 8, 7), mode='trilinear'), #3x4x3
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(hidden_dim),
            nn.Upsample(size=(14, 17, 14), mode='trilinear'), #3x4x3
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(hidden_dim),
            nn.Upsample(size=(29, 35, 29), mode='trilinear'), #3x4x3
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(hidden_dim),
            nn.Upsample(size=(58, 70, 58), mode='trilinear'), #3x4x3
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(hidden_dim),
            nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.model(x)