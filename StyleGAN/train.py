import torch
from torch import nn, optim
from torchvision.datasets import SVHN
from torchvision import transforms
from torch.utils import data
from torch.nn.functional import interpolate
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from mri_style_gan import StyleGenerator, ProgressiveDiscriminator
from losses import compute_gradient_penalty, fake_loss, real_loss
import numpy as np

### Initialization

z_size = 128
hidden_size = 256
start_resolution = 8
batch_size = 16
max_resolution = 64

make_plots = True


G = StyleGenerator(hidden_size, z_size, start_resolution=start_resolution).cuda()
D = ProgressiveDiscriminator(hidden_size, start_resolution=start_resolution).cuda()

loss_func = "WGAN-GP"

# Optimizers
betas = [0, 0.99]
lr = 0.002

d_optimizer = optim.Adam(D.parameters(), betas=betas, lr=lr)
g_optimizer = optim.Adam(G.parameters(), betas=betas, lr=lr)


def get_resizing_func(size):
    return lambda img: interpolate(
        img,
        size=(size, size, size),
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )


resize_to = {size: get_resizing_func(size) for size in range(4, max_resolution + 1, 4)}


labels = np.load("../data/genders.npy")
tensors = np.load("../data/scans.npy")

scan_data = tensors[:, None, :, 3:-3]
scan_data = torch.nn.functional.pad(torch.Tensor(scan_data), (3, 3, 0, 0, 3, 3))

ds = data.TensorDataset(scan_data, torch.Tensor(labels))
train_loader = data.DataLoader(ds, batch_size=batch_size)



# training hyperparams
num_epochs = 100
start_epoch = 0
# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 31
increase_resolution_every = 8
increase_alpha_every = len(ds) // batch_size * 3 // 10

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=4
fixed_z = torch.randn(sample_size, z_size, dtype=torch.float32).cuda()

cur_resolution = start_resolution
lambda_gp = 10

data_max, data_min = tensors.max(), tensors.min()
data_range = data_max - data_min

# train the network
D.train()
G.train()

for epoch in range(start_epoch, num_epochs):
    
    for batch_i, (real_images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
                
        bs = real_images.size(0)
        if bs < 3:
          continue
        if np.prod(real_images.shape) == 0:
          continue

        real_images = real_images.cuda()
        real_images = (real_images - data_min) / data_range * 2 - 1
        if cur_resolution > 4:
          down_up = interpolate(resize_to[cur_resolution // 2](real_images), mode='trilinear', scale_factor=2, )
          down = resize_to[cur_resolution](real_images)
          real_images = resize_to[cur_resolution](real_images) * D.alpha + \
               down_up * (1 - D.alpha)
        else:
          real_images = resize_to[cur_resolution](real_images)
        
        d_optimizer.zero_grad()
        
        # 1. Train with real images

        D_real = D(real_images)
        
        # 2. Train with fake images
        
        with torch.no_grad():
            z = torch.randn(bs, z_size, dtype=torch.float32).cuda()
            fake_images = G(z)
              
        D_fake = D(fake_images)
        
        # add up loss and perform backprop
        if loss_func == "WGAN-GP":
          gradient_penalty = compute_gradient_penalty(D, real_images.data, fake_images.data)
          d_loss = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gradient_penalty + 0.001 * torch.mean(D_real**2)
        else:
          d_real_loss = real_loss(D_real, smooth=True)
          d_fake_loss = fake_loss(D_fake)
          d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()
        
        
        g_optimizer.zero_grad()
        
        # 1. Train with fake images and flipped labels
        
        z = torch.randn(bs, z_size, dtype=torch.float32).cuda()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_fake = D(fake_images)
        if loss_func == "WGAN-GP":
          g_loss = -torch.mean(D_fake)
        else:
          g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # perform backprop
        g_loss.backward()
        g_optimizer.step()


        if (batch_i + 1) % increase_alpha_every == 0:
          G.increase_alpha()
          D.increase_alpha()


        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))

    if (epoch + 1) % increase_resolution_every == 0 and cur_resolution < max_resolution:
      cur_resolution = cur_resolution * 2
      G.increase_resolution()
      D.increase_resolution()
      increase_resolution_every += 1 + increase_resolution_every
      batch_size //= 2
      train_loader = data.DataLoader(ds, batch_size=batch_size)
      print(f"Increasing image resolution to {cur_resolution}")
    
    # generate and save sample, fake images
    G.eval() # eval mode for generating samples
    samples_z = G(fixed_z)
    G.train() # back to train mode

    if make_plots:
      f = plt.figure(figsize=(24, 4))
      for i in range(1, 4):
        ax = plt.subplot(1, 4, i)
        plot_scan(samples_z[i-1][0].detach().cpu(), axes=ax)
      ax = plt.subplot(1, 4, 4)
      plot_scan(real_images[0][0].detach().cpu(), axes=ax)
      plt.show()

    torch.save(G.state_dict(), f'checkpoints/stylegan_generator_weights_{epoch}.pkl')
    torch.save(D.state_dict(), f'checkpoints/stylegan_discriminator_weights_{epoch}.pkl')
