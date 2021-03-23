import torch
from torch import nn
from torch.utils import data
from models import VAE3D, AE3D
from torch import optim
from losses import vae_loss
import numpy as np

### Initialization

make_plots = True
model_type = "VAE"
lr = 0.002

if model_type == "VAE":
    model = VAE3D()
else:
    model = AE3D()

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()


labels = np.load("../data/genders.npy")
tensors = np.load("../data/scans.npy")

scan_data = tensors[:, None]

ds = data.TensorDataset(scan_data, torch.Tensor(labels))
train_loader = data.DataLoader(ds, batch_size=batch_size)



data_max, data_min = tensors.max(), tensors.min()
data_range = data_max - data_min

for e in tqdm(range(100)):
    running_loss = 0
    for i, (X, y) in tqdm(enumerate(train_loader)):
        X = X.cuda()
        y = y.cuda()

        pred = model(X)
        if model_type == "VAE":
            means, stds, outs = pred
            kl_loss, recon_loss = vae_loss(means, stds, outs, X, beta=0.8)
            loss = kl_loss + recon_loss
        else:
            loss = criterion(pred, X)

        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            torch.save(model.state_dict(), "checkpoints/model_vae_KL08_HR.pth")

            if model_type == "VAE":
                print("Reconstruction Loss: ", recon_loss.item())
                print("KL_divergence: ", kl_loss.item())
            else:
                print("MSE Loss: ", loss.item())
            if make_plots:
                affine = np.diag([1, 1, 1, 1])
                t1 = nib.Nifti1Image(X[0][0].detach().cpu().numpy(), affine)
                t2 = nib.Nifti1Image(outs[0][0].detach().cpu().numpy(), affine)

                plotting.plot_img(t1, [s / 2 for s in t1.shape], title="Truth")
                plotting.plot_img(t2, [s / 2 for s in t2.shape], title="Reconstructed")
                plt.show()
