import torch
from torch import nn



class AE3D(nn.Module):
  def __init__(self, hidden_dim=256):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv3d(1, 64, kernel_size=3, padding=1), #58, 70, 58
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.MaxPool3d(2), #29, 35, 29
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.MaxPool3d(2), #14, 17, 14
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.MaxPool3d(2), #7, 8, 7
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.MaxPool3d(2), #3, 4, 3
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, hidden_dim, kernel_size=3, padding=1), # 3, 4, 3
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten()
    )

    self.decoder = nn.Sequential(
        nn.Unflatten(1, (hidden_dim, 1, 1, 1)), # 1x1x1
        nn.Upsample(size=(3, 4, 3), mode='trilinear',), #3x4x3
        nn.Conv3d(hidden_dim, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Upsample(size=(7, 8, 7), mode='trilinear'), #3x4x3
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Upsample(size=(14, 17, 14), mode='trilinear'), #3x4x3
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Upsample(size=(29, 35, 29), mode='trilinear'), #3x4x3
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Upsample(size=(58, 70, 58), mode='trilinear'), #3x4x3
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 1, kernel_size=3, padding=1)
    )
  def forward(self, X):
    x = self.encoder(X)
    return self.decoder(x)



class VAE3D(nn.Module):
  def __init__(self, hidden_dim=256, latent_space=64):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv3d(1, 256, kernel_size=3, padding=1), # 64,58, 70, 58
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, 256, kernel_size=3, padding=1), #64, 29, 35, 29
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.MaxPool3d(2), #64, 14, 17, 14
        nn.Conv3d(256, 256, kernel_size=3, padding=1), #64, 14, 17, 14
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.MaxPool3d(2), #64, 7, 8, 7
        nn.Conv3d(256, 256, kernel_size=3, padding=1), #64, 7, 8, 7
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.MaxPool3d(2), #64, 3, 4, 3
        nn.Conv3d(256, 256, kernel_size=3, padding=1), #64, 3, 4, 3
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, hidden_dim, kernel_size=3, padding=1), #hidden_dim, 3, 4, 3
        nn.AdaptiveAvgPool3d((1, 1, 1)), #hidden_dim, 1, 1, 1
        nn.Flatten() # hidden_dim
    )
    self.encoder_means = nn.Linear(hidden_dim, latent_space)
    self.encoder_stds = nn.Linear(hidden_dim, latent_space)

    self.decoder = nn.Sequential(
        nn.Unflatten(1, (latent_space, 1, 1, 1)), # 1x1x1
        nn.Upsample(size=(3, 4, 3), mode='trilinear',), #3x4x3
        nn.Conv3d(latent_space, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Upsample(size=(7, 8, 7), mode='trilinear'), #3x4x3
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Upsample(size=(14, 17, 14), mode='trilinear'), #3x4x3
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Upsample(size=(29, 35, 29), mode='trilinear'), #3x4x3
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, 1, kernel_size=3, padding=1)
    )

  def encode(self, inputs):
    inputs = self.encoder(inputs)
    means = self.encoder_means(inputs)
    stds = self.encoder_stds(inputs)
    return means, stds
  
  def decode(self, z):
    out = self.decoder(z)
    return out
  
  def sample(self, means, stds):
    epsilon = torch.randn(means.shape).to(device=cuda)
    z = means + epsilon * stds
    return z
  
  def forward(self, inputs):
    means, stds = self.encode(inputs)
    z = self.sample(means, stds)
    out = self.decode(z)
    

    return means, stds, out     