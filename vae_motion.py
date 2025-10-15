# vae_motion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):
    def __init__(self, in_dim=2, hid=128, z_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)

    def forward(self, x):
        # x: (B, T, 2)
        _, h = self.gru(x)    # h: (num_layers, B, hid)
        h = h[-1]             # (B, hid)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

class GRUDecoder(nn.Module):
    def __init__(self, z_dim=32, out_dim=2, hid=128, obs_h=20):
        super().__init__()
        self.obs_h = obs_h
        self.fc0 = nn.Linear(z_dim, hid)
        # reconstruct past as a sequence
        self.gru = nn.GRU(out_dim, hid, batch_first=True)
        self.recon_head = nn.Linear(hid, out_dim)
        # next-step and speed heads
        self.next_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 2)
        )
        self.speed_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, z):
        # Seed decoder with zeros and condition on z via initial hidden state
        B = z.size(0)
        h0 = torch.tanh(self.fc0(z)).unsqueeze(0)  # (1, B, hid)
        y0 = torch.zeros(B, self.obs_h, 2, device=z.device)  # teacher-forcing free decoder (zeros)

        out, _ = self.gru(y0, h0)           # (B, T, hid)
        recon = self.recon_head(out)        # (B, T, 2)
        # Use last hidden feature to predict next-step and speed
        feat_last = out[:, -1, :]           # (B, hid)
        next_delta = self.next_head(feat_last)   # (B, 2)
        speed = self.speed_head(feat_last)       # (B, 1)
        return recon, next_delta, speed

class VAE(nn.Module):
    def __init__(self, in_dim=2, obs_h=20, hid=128, z_dim=32):
        super().__init__()
        self.encoder = GRUEncoder(in_dim=in_dim, hid=hid, z_dim=z_dim)
        self.decoder = GRUDecoder(z_dim=z_dim, out_dim=in_dim, hid=hid, obs_h=obs_h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        recon, next_delta, speed = self.decoder(z)
        return recon, next_delta, speed, mu, logvar
