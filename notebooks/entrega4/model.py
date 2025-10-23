import torch
import torch.nn.functional as F
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        #encoder
        self.img_hid = nn.Linear(input_dim, hidden_dim)
        self.hid_mu = nn.Linear(hidden_dim, latent_dim) 
        self.hid_sigma = nn.Linear(hidden_dim, latent_dim) 
        
        #decoder
        self.z_hid = nn.Linear(latent_dim, hidden_dim)
        self.hid_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.img_hid(x))
        return self.hid_mu(h), self.hid_sigma(h)

    def decode(self, z):
        h = F.relu(self.z_hid(z))
        return torch.sigmoid(self.hid_out(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return self.decode(z), mu, sigma
    
if __name__ == "__main__":
    x = torch.randn(16, 28*28)
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=40) 
