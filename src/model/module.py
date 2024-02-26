import torch
import torch.nn as nn

from utils.module_utils import View


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, latent_size: int = 256, dropout: float = 0.2, batch_norm: bool = True):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input size: 3 x 128 x 128
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # Output size: 32 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(32) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # Output size: 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Output size: 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # Output size: 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # Output size: 512 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(512) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.Flatten(),  # Output size: 8192
            nn.Linear(8192, latent_size * 2)  # Output size: 512
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8192),  # Output size: 8192
            nn.ReLU(),
            nn.BatchNorm1d(8192) if batch_norm else None,
            nn.Dropout(dropout) if dropout > 0 else None,
            View((-1, 512, 4, 4)),  # Output size: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # Output size: 256 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(256) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Output size: 128 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Output size: 64 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # Output size: 32 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(32) if batch_norm else None,
            nn.Dropout2d(dropout) if dropout > 0 else None,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)  # Output size: 3 x 128 x 128
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.tanh(self.decoder(z))
    
    def sample(self, num_samples: int = 1):
        z = torch.randn(num_samples, self.latent_size)
        return self.decode(z)
    
    def distributionize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mu, std)
    
    def get_embedding(self, x, return_distribution: bool = False):
        mu, logvar = self.encode(x)        
        if return_distribution:
            return self.distributionize(mu, logvar)
        
        z = self.reparameterize(mu, logvar)
        return z