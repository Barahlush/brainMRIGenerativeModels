import torch


criterion = torch.nn.MSELoss()
def vae_loss(means, stds, outs, X, beta=0.8)
    squared_stds = stds ** 2
    reconstruction_loss = criterion(outs, X)
    KL_divergence = torch.mean(
        -0.5
        * torch.sum(1 + torch.log(squared_stds) - means ** 2 - squared_stds, dim=1),
        dim=0,
    )
    KL_divergence = KL_divergence * beta
    return KL_divergence, reconstruction_loss