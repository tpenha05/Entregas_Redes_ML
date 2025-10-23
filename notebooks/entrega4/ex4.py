import torch
import os
from torch import nn
from model import VAE
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 28*28
HIDDEN_DIM = 400
LATENT_DIM = 40

EPOCHS = 50

BATCH_SIZE = 64

LR = 1e-4

CHECKPOINT_DIR = "./checkpoints"
SAMPLES_DIR = "./samples"
RECONS_DIR = "./reconstructions"
PLOTS_DIR = "./plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RECONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True) #.ToTensor() transforma a imagem para tensores e normaliza entre 0 e 1
test_ds = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.15, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss = nn.BCELoss(reduction='sum')


def save_reconstruction(original, recon, epoch, n=8):
    if original.dim() == 2:
        original = original.view(-1, 1, 28, 28)
    if recon.dim() == 2:
        recon = recon.view(-1, 1, 28, 28)
    orig_grid = make_grid(original[:n], nrow=n, padding=2)
    recon_grid = make_grid(recon[:n], nrow=n, padding=2)
    stacked = torch.cat([orig_grid, recon_grid], dim=1)
    save_image(stacked, os.path.join(RECONS_DIR, f"recon_epoch{epoch}.png"))

def save_samples(epoch, n=64):
    z = torch.randn(n, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        samples = model.decode(z)

        if samples.dim() == 2:
            samples = samples.view(-1, 1, 28, 28)
    save_image(samples.cpu(), os.path.join(SAMPLES_DIR, f"samples_epoch{epoch}.png"), nrow=8)


for epoch in range(EPOCHS):
    model.train()
    train_total = 0.0
    train_bce = 0.0
    train_kld = 0.0

    for xb, yb in train_loader:
        xb = xb.view(-1, INPUT_DIM).to(DEVICE)
        
        recon_x, mu, sigma = model(xb)
        
        bce_loss = loss(recon_x, xb)
        kld_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        
        loss_total = bce_loss + kld_loss
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        train_total += loss_total.item()
        train_bce += bce_loss.item()
        train_kld += kld_loss.item()

    avg_loss = train_total / len(train_dataset)
    avg_bce = train_bce / len(train_dataset)
    avg_kld = train_kld / len(train_dataset)
    print(f"[Epoch {epoch}/{EPOCHS}] Train: total={avg_loss:.6f}, BCE={avg_bce:.6f}, KLD={avg_kld:.6f}")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"vae_epoch{epoch}.pt"))

    model.eval()
    with torch.no_grad():
        for xb_val, _ in val_loader:
            xb_val_flat = xb_val.view(-1, INPUT_DIM).to(DEVICE)
            recon_val, _, _ = model(xb_val_flat)
            save_reconstruction(xb_val, recon_val.cpu(), epoch)
            break
        save_samples(epoch)

    val_total = 0.0
    with torch.no_grad():
        for xb_val, _ in val_loader:
            xb_val = xb_val.view(-1, INPUT_DIM).to(DEVICE)
            recon_val, mu_val, sigma_raw_val = model(xb_val)
            bce_v = loss(recon_val, xb_val)
            kld_v = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
            val_total += (bce_v + kld_v).item()

    print(f"Validation loss/sample: {val_total / len(val_dataset):.6f}")

model.eval()
test_total = 0.0
with torch.no_grad():
    for xb_test, _ in test_loader:
        xb_test = xb_test.view(-1, INPUT_DIM).to(DEVICE)
        recon_test, mu_test, sigma_raw_test = model(xb_test)
        bce_t = loss(recon_test, xb_test)
        kld_t = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        test_total += (bce_t + kld_t).item()

print(f"Final test loss/sample: {test_total / len(test_ds):.6f}")

mus, labels = [], []
with torch.no_grad():
    for xb_val, y_val in val_loader:
        xb_flat = xb_val.view(-1, INPUT_DIM).to(DEVICE)
        _, mu_val, _ = model(xb_flat)
        mus.append(mu_val.cpu())
        labels.append(y_val)
mus = torch.cat(mus, dim=0).numpy()
labels = torch.cat(labels, dim=0).numpy()

pca = PCA(n_components=2)
mus_pca = pca.fit_transform(mus)
plt.figure(figsize=(6,6))
plt.scatter(mus_pca[:,0], mus_pca[:,1], c=labels, cmap='tab10', s=6)
plt.title("Latent space PCA (2D)")
plt.colorbar()
plt.savefig(os.path.join(PLOTS_DIR, "latent_pca.png"))
plt.close()

print("Treinamento concluído. Verifique as pastas: checkpoints, reconstructions, samples, plots.")