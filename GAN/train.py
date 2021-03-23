from torch import nn, optim
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import Generator, Discriminator
from losses import fake_loss, real_loss

labels = np.load("../data/genders.npy")
tensors = np.load("../data/scans.npy")

scan_data = tensors[:, None]

ds = TensorDataset(torch.Tensor(tensors[:, None]), torch.Tensor(labels))
train_loader = DataLoader(ds, batch_size=8)

lr = 0.002
beta1=0.5
beta2=0.999 # default value

z_size = 64
train_loader = loader

D = Discriminator(z_size)
G = Generator(z_size)
gain = nn.init.calculate_gain('leaky_relu', 0.2)
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=gain)
        m.bias.data.fill_(0.001)
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_normal_(m.weight, gain=gain)
        m.bias.data.fill_(0.001)
D.apply(init_weights)
G.apply(init_weights)

def scale(x, max, min, feature_range=(-1, 1)):
    out_min, out_max = feature_range
    x = (x - min) / (max - min) * (out_max - out_min) + out_min
    return x

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')


d_optimizer = optim.Adam(D.parameters(), lr=5e-5, betas=[beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr=5e-5, betas=[beta1, beta2])

num_epochs = 10
samples = []
losses = []
print_every = 500


data_max, data_min = tensors.max(), tensors.min()

# train the network
for epoch in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        # important rescaling step
        real_images = scale(real_images, data_max, data_min)

        # Compute the discriminator losses on real images 
        if train_on_gpu:
            real_images = real_images.cuda()
        real_images.requires_grad_(False)

        D_real = D(real_images)
        d_real_loss = real_loss(D_real, smooth=True)
        
        # 2. Train with fake images
        # Generate fake images
        z = torch.randn(batch_size, z_size).cuda().float()
        z.requires_grad_(False)
        # print(z.shape)
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images            
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)
        
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        # d_loss = -d_real_loss - d_fake_loss

        # d_optimizer.step()

        fake_correct = ((torch.sigmoid(D_fake) > 0.5) == 0).sum()
        real_correct = ((torch.sigmoid(D_real) > 0.5) == 1).sum()
        d_loss.backward()
        if d_loss.item() > 1e-4:
          d_optimizer.step()
        d_optimizer.zero_grad()

        
        # 1. Train with fake images and flipped labels
        # Generate fake images
        z = torch.randn(batch_size, z_size).cuda().float()

        if train_on_gpu:
            z = z.cuda()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images 
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        # g_loss = fake_loss(D_fake)
        
        # perform backprop
        g_loss.backward()
        if g_loss.item() > 1e-4:
          g_optimizer.step()
        g_optimizer.zero_grad()

        if batch_i%50 == 0:
        # if batch_i%n_critic == 0:
          # clear_output()
          print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

          if make_plots:
            affine = np.diag([1, 1, 1, 1])
            t1 = nib.Nifti1Image(real_images[0][0].detach().cpu().numpy(), affine)
            t2 = nib.Nifti1Image(fake_images[0][0].detach().cpu().numpy(), affine)

            plotting.plot_img(t1,[s/2 for s in t1.shape], title="Truth")
            plotting.plot_img(t2,[s/2 for s in t2.shape], title="Reconstructed")

            plt.show()
        del real_images
    
