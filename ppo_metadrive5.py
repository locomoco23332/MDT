"""
PPO Algorithm Implementation for MetaDrive Environment

This module implements the Proximal Policy Optimization (PPO) algorithm
for training agents in the MetaDrive autonomous driving environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import typing
from typing import List, Tuple, Callable, Optional
import gymnasium as gym
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from metadrive.envs.top_down_env import TopDownMetaDrive

# Register MetaDrive environment
gym.register(id="MetaDrive-topdown", entry_point=TopDownMetaDrive, kwargs=dict(config={}))

# Disable logging from metadrive
logging.getLogger("metadrive.envs.base_env").setLevel(logging.WARNING)
class DWConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class CNN8420x2_Lite(nn.Module):
    """
    Input:  [B,5,84,84]
    Output: [B,20,2]
    """
    def __init__(self, out_steps=20, out_dim=2, width_mult=1.0, use_gn=False):
        super().__init__()
        self.out_steps, self.out_dim = out_steps, out_dim
        C1 = int(16 * width_mult)
        C2 = int(32 * width_mult)
        C3 = int(64 * width_mult)

        Norm = (lambda c: nn.GroupNorm(1, c)) if use_gn else nn.BatchNorm2d

        # stem: 살짝만 확장
        self.stem = nn.Sequential(
            nn.Conv2d(5, C1, 3, stride=2, padding=1, bias=False),   # 84->42
            Norm(C1), nn.ReLU(inplace=True),
        )
        # depthwise separable stages
        self.stage2 = DWConvBlock(C1, C2, stride=2)                 # 42->21
        self.stage3 = DWConvBlock(C2, C3, stride=2)                 # 21->11
        self.pool = nn.AdaptiveAvgPool2d(1)                         # -> [B,C3,1,1]

        # 1x1 head to 40 (=20*2)
        self.head = nn.Conv2d(C3, out_steps * out_dim, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)            # [B,C1,42,42]
        x = self.stage2(x)          # [B,C2,21,21]
        x = self.stage3(x)          # [B,C3,11,11]
        x = self.pool(x)            # [B,C3,1,1]
        x = self.head(x)            # [B,40,1,1]
        x = x.squeeze(-1).squeeze(-1)            # [B,40]
        return x.view(x.size(0), self.out_steps, self.out_dim)  # [B,20,2]
class CNN8420x2(nn.Module):
    """
    Input:  [B, 5, 84, 84]
    Output: [B, 20, 2]
    """
    def __init__(self, out_steps=20, out_dim=2):
        super().__init__()
        self.out_steps = out_steps
        self.out_dim = out_dim
        out_channels = out_steps * out_dim  # 40

        # Feature extractor: downsample 84->42->21->11->6 (stride=2)
        self.backbone = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),  # [B,32,42,42]
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B,64,21,21]
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [B,128,11,11]
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# [B,256,6,6]
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)) # [B,256,1,1]
        )

        # 1x1 conv head to get exactly out_steps*out_dim channels (40)
        self.head = nn.Conv2d(256, out_channels, kernel_size=1)  # [B,40,1,1]

    def forward(self, x):
        # x: [B,5,84,84]
        feats = self.backbone(x)          # [B,256,1,1]
        y = self.head(feats)              # [B,40,1,1]
        y = y.squeeze(-1).squeeze(-1)     # [B,40]
        y = y.view(x.size(0), self.out_steps, self.out_dim)  # [B,20,2]
        return y
import torch
import torch.nn as nn

class DWSep2(nn.Module):
    def __init__(self, cin, cout, stride=1, use_gn=True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(1, cout) if use_gn else nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.norm(x)
        return self.act(x)
class CNN840_Micro(nn.Module):
    """
    Input:  [B, 5, 84, 84]
    Output: [B, 20, 2]  (총 40차원)
    단계: 84->42->21 + GAP + 1x1 Head(40ch)
    """
    def __init__(self, out_steps=20, out_dim=2, base_ch=24, mid_ch=48, use_gn=True):
        super().__init__()
        self.out_steps, self.out_dim = out_steps, out_dim

        # 84 -> 42 (얕은 stem)
        self.stem = nn.Sequential(
            nn.Conv2d(5, base_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, base_ch) if use_gn else nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        # 42 -> 21 (DW Separable)
        self.stage = DWSep2(base_ch, mid_ch, stride=2, use_gn=use_gn)

        # GAP -> 1x1 Head(40ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(mid_ch, out_steps * out_dim, 1, bias=True)  # 40ch

    def forward(self, x):
        x = self.stem(x)          # [B, base_ch, 42, 42]
        x = self.stage(x)         # [B, mid_ch, 21, 21]
        x = self.pool(x)          # [B, mid_ch, 1, 1]
        x = self.head(x).flatten(1)      # [B, 40]
        return x.view(-1, self.out_steps, self.out_dim)  # [B, 20, 2]

class Decoder40to5x84x84(nn.Module):
    """
    Input:  [B, 20, 2]  -> flattened to [B, 40]
    Output: [B, 5, 84, 84]
    Mirror of encoder: 6->11->21->42->84 via ConvTranspose2d
    """
    def __init__(self, seed_channels=256, seed_hw=6, out_ch=5):
        super().__init__()
        self.seed_c = seed_channels
        self.seed_hw = seed_hw
        seed_size = seed_channels * seed_hw * seed_hw  # 256*6*6 = 9216

        # 1) vector(40) -> seed feature map [B,256,6,6]
        self.fc = nn.Sequential(
            nn.Linear(40, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, seed_size),
            nn.ReLU(inplace=True),
        )

        # 2) 6 -> 11  (k=3, s=2, p=1, op=0) => 11
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(seed_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # 3) 11 -> 21 (k=3, s=2, p=1, op=0) => 21
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # 4) 21 -> 42 (k=3, s=2, p=1, op=1) => 42
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # 5) 42 -> 84 (k=3, s=2, p=1, op=1) => 84
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 출력 범위가 필요하면 여기서 activation 선택:
            # nn.Tanh()  # [-1,1] 범위
            # nn.Sigmoid()  # [0,1] 범위
        )

    def forward(self, z):               # z: [B,20,2]
        B = z.size(0)
        z = z.view(B, -1)               # [B,40]
        seed = self.fc(z)               # [B, 9216]
        seed = seed.view(B, self.seed_c, self.seed_hw, self.seed_hw)  # [B,256,6,6]
        x = self.deconv1(seed)          # [B,256,11,11]
        x = self.deconv2(x)             # [B,128,21,21]
        x = self.deconv3(x)             # [B, 64,42,42]
        x = self.deconv4(x)             # [B,  5,84,84]
        return x
class Decoder40to5x84x84_Lite(nn.Module):
    """
    Input:  [B, 20, 2] -> flatten [B, 40]
    Output: [B, 5, 84, 84]
    경량: 40 -> (C*21*21) 투영 -> 업샘플 x2 x2
    """
    def __init__(self, out_ch=5, base_ch=64, hidden=128, act='relu', norm='bn', out_activation=None):
        super().__init__()
        self.H0 = self.W0 = 21
        self.C0 = base_ch
        self.out_activation = out_activation

        Act = nn.ReLU(inplace=True) if act == 'relu' else nn.GELU()
        def Norm(c):
            if norm == 'gn': return nn.GroupNorm(1, c)     # 배치 작을 때 안정
            return nn.BatchNorm2d(c)

        # 1) 40 -> C0*21*21  (FC를 확 줄임: 40->hidden->C0*21*21)
        self.fc = nn.Sequential(
            nn.Linear(40, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.C0 * self.H0 * self.W0),
            nn.ReLU(inplace=True),
        )

        # 업블록: Upsample(×2) + Conv3x3 ×2
        def up_block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),       # 21->42, 42->84
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                Norm(cout), Act,
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                Norm(cout), Act,
            )

        # 21 -> 42 -> 84
        self.up1 = up_block(self.C0, self.C0 // 2)  # 64 -> 32
        self.up2 = up_block(self.C0 // 2, self.C0 // 4)  # 32 -> 16

        # 헤드
        self.head = nn.Conv2d(self.C0 // 4, out_ch, 3, padding=1)

    def forward(self, z):  # z: [B,20,2]
        B = z.size(0)
        z = z.view(B, -1)                               # [B,40]
        seed = self.fc(z).view(B, self.C0, self.H0, self.W0)  # [B,64,21,21]
        x = self.up1(seed)                              # [B,32,42,42]
        x = self.up2(x)                                 # [B,16,84,84]
        y = self.head(x)                                # [B, 5,84,84]
        if self.out_activation is not None:
            y = self.out_activation(y)                  # Tanh / Sigmoid 등
        return y
import torch
import torch.nn as nn

class DWSep(nn.Module):
    def __init__(self, cin, cout, act=True, gn=True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(1, cout) if gn else nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)

class Decoder40to5x84x84_Micro(nn.Module):
    """
    Input:  [B,20,2] -> [B,40]
    Seed:   [B, C0, 21, 21]
    Upscale: PixelShuffle x2 -> 42, PixelShuffle x2 -> 84
    Head:   5ch
    """
    def __init__(self, out_ch=5, base_ch=48, hidden=96, use_gn=True, out_activation=None):
        super().__init__()
        self.H0 = self.W0 = 21
        self.C0 = base_ch
        self.out_activation = out_activation

        # FC를 더 줄임: 40 -> hidden -> C0*21*21
        self.fc = nn.Sequential(
            nn.Linear(40, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.C0 * self.H0 * self.W0),
            nn.ReLU(inplace=True),
        )

        # PixelShuffle 업블록: conv로 채널을 (r^2 * target_ch)로 만들고 PixelShuffle(r)
        # 21 -> 42
        self.pre_shuffle1 = nn.Conv2d(self.C0, (2*2) * (self.C0 // 2), 1, bias=False)
        self.shuffle1 = nn.PixelShuffle(2)            # [B, C0//2, 42, 42]
        self.refine1 = DWSep(self.C0 // 2, self.C0 // 2, act=True, gn=use_gn)

        # 42 -> 84
        self.pre_shuffle2 = nn.Conv2d(self.C0 // 2, (2*2) * (self.C0 // 4), 1, bias=False)
        self.shuffle2 = nn.PixelShuffle(2)            # [B, C0//4, 84, 84]
        self.refine2 = DWSep(self.C0 // 4, self.C0 // 4, act=True, gn=use_gn)

        # Head
        self.head = nn.Conv2d(self.C0 // 4, out_ch, 3, padding=1)

    def forward(self, z):   # z: [B,20,2]
        B = z.size(0)
        z = z.view(B, -1)                                     # [B,40]
        x = self.fc(z).view(B, self.C0, self.H0, self.W0)     # [B,C0,21,21]

        x = self.pre_shuffle1(x)
        x = self.shuffle1(x)                                  # [B,C0//2,42,42]
        x = self.refine1(x)

        x = self.pre_shuffle2(x)
        x = self.shuffle2(x)                                  # [B,C0//4,84,84]
        x = self.refine2(x)

        y = self.head(x)                                      # [B,5,84,84]
        if self.out_activation is not None:
            y = self.out_activation(y)                        # Tanh / Sigmoid 등
        return y
import torch
import torch.nn as nn
import math
class ActionDecoder(nn.Module):
    def __init__(self, z_dim=40, out_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim)
        )
    def forward(self, z):           # z can be [B,40] or [B,20,2]
        if z.dim() == 3:
            z = z.view(z.size(0), -1)   # → [B,40]
        return self.net(z)
class Decoder40to5x84x84_LinearOnly(nn.Module):
    """
    Conv 없이 Linear만 사용한 단순 MLP 디코더
    Input:  [B,20,2] -> flatten [B,40]
    Output: [B,5,84,84] (35280차원)
    - 큰 배치에서 메모리 절약을 위해 chunked 출력 지원
    """
    def __init__(self, hidden_dims=(512, 1024), out_activation=None, chunk_size=8192):
        super().__init__()
        self.in_dim = 40
        self.out_ch, self.H, self.W = 5, 84, 84
        self.out_dim = self.out_ch * self.H * self.W  # 35280
        self.out_activation = out_activation
        self.chunk_size = chunk_size  # 마지막 fc를 청크로 나눠 계산 (메모리 절약)

        layers = []
        prev = self.in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        # 마지막 FC는 big tensor이므로 별도 보관(청크 분할 계산)
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, self.out_dim)

    def forward(self, z):  # z: [B,20,2]
        B = z.size(0)
        x = z.view(B, -1)           # [B,40]
        h = self.mlp(x)             # [B,Hid]

        # ----- 메모리 절약: fc_out을 청크로 나눠서 계산 -----
        if self.chunk_size is None or self.chunk_size >= self.out_dim:
            y = self.fc_out(h)      # [B,35280]
        else:
            outs = []
            start = 0
            while start < self.out_dim:
                end = min(start + self.chunk_size, self.out_dim)
                # weight/slice만 가져와 matmul (Linear와 동일)
                W = self.fc_out.weight[start:end, :]   # [chunk, Hid]
                b = self.fc_out.bias[start:end]        # [chunk]
                outs.append(h @ W.t() + b)             # [B,chunk]
                start = end
            y = torch.cat(outs, dim=1)                 # [B,35280]
        # -----------------------------------------------

        if self.out_activation is not None:
            y = self.out_activation(y)
        return y.view(B, self.out_ch, self.H, self.W)  # [B,5,84,84]
class ActionToStateVAE(nn.Module):
    """
    Action -> State VAE
    Input: [B, 2] (throttle, steering)
    Output: [B, 5, 84, 84] (state reconstruction)
    Latent: [B, 40] (same dimension as state->action VAE)
    """
    def __init__(self, latent_dim=40, recon_activation=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Action encoder: [B, 2] -> [B, 40]
        self.action_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
        )
        
        # μ, logσ² 생성 헤드 (40 -> 40)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        
        # State decoder: [B, 40] -> [B, 5, 84, 84]
        self.state_decoder = Decoder40to5x84x84_LinearOnly(
            hidden_dims=(512, 1024), 
            out_activation=recon_activation
        )
    
    @staticmethod
    def reparameterize(mu, logvar):
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, action):
        """
        action: [B, 2]
        return: mu[B, 40], logvar[B, 40], z[B, 40]
        """
        h = self.action_encoder(action)  # [B, 40]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
    
    def decode(self, z):
        """
        z: [B, 40] -> [B, 5, 84, 84]
        """
        # z를 [B, 20, 2] 형태로 reshape
        z_reshaped = z.view(z.size(0), 20, 2)
        recon = self.state_decoder(z_reshaped)
        return recon
    
    def forward(self, action, return_latent=False):
        """
        action: [B, 2]
        return: recon[B, 5, 84, 84], mu[B, 40], logvar[B, 40], (optional) z[B, 40]
        """
        mu, logvar, z = self.encode(action)
        recon = self.decode(z)
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar, z
    
    def loss(self, target_state, recon, mu, logvar, beta=1.0, recon_type="mse"):
        """
        VAE loss = ReconLoss + beta * KL
        """
        if recon_type.lower() == "mse":
            recon_loss = F.mse_loss(recon, target_state, reduction="mean")
        elif recon_type.lower() == "bce":
            recon_loss = F.binary_cross_entropy(recon, target_state, reduction="mean")
        else:
            raise ValueError("recon_type must be 'mse' or 'bce'")
        
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total = recon_loss + beta * kl
        return total, recon_loss, kl


class DualVAEWithLatentFusion(nn.Module):
    """
    Dual VAE with Latent Fusion
    - State->Action VAE: [B, 5, 84, 84] -> [B, 2] (action)
    - Action->State VAE: [B, 2] -> [B, 5, 84, 84] (state)
    - Latent Fusion: Concatenate both latents and project back to original size
    """
    def __init__(self, latent_dim=40, recon_activation=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # State->Action VAE (기존)
        self.state_to_action_vae = ImageVAE_CNN840Micro_LinearDec(recon_activation=recon_activation)
        
        # Action->State VAE (새로 추가)
        self.action_to_state_vae = ActionToStateVAE(latent_dim=latent_dim, recon_activation=recon_activation)
        
        # Latent fusion: concatenated latent (80) -> original latent (40)
        self.latent_fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),  # 80 -> 80
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),      # 80 -> 40
            nn.LayerNorm(latent_dim)
        )
    
    def forward_state_to_action(self, state, return_latent=False):
        """
        State -> Action VAE forward pass
        state: [B, 5, 84, 84]
        return: action[B, 2], mu[B, 40], logvar[B, 40], (optional) z[B, 40]
        """
        return self.state_to_action_vae(state, return_latent=return_latent)
    
    def forward_action_to_state(self, action, return_latent=False):
        """
        Action -> State VAE forward pass
        action: [B, 2]
        return: state_recon[B, 5, 84, 84], mu[B, 40], logvar[B, 40], (optional) z[B, 40]
        """
        return self.action_to_state_vae(action, return_latent=return_latent)
    
    def forward_with_fusion(self, state, action, return_latents=False):
        """
        Forward pass with latent fusion
        state: [B, 5, 84, 84]
        action: [B, 2]
        return: fused_latent[B, 40], state_action[B, 2], state_recon[B, 5, 84, 84]
        """
        # State -> Action VAE
        state_action, state_mu, state_logvar, state_z = self.state_to_action_vae(state, return_latent=True)
        
        # Action -> State VAE
        action_state_recon, action_mu, action_logvar, action_z = self.action_to_state_vae(action, return_latent=True)
        
        # Latent fusion: concatenate both latents
        concatenated_latent = torch.cat([state_z, action_z], dim=1)  # [B, 80]
        
        # Project back to original latent size
        fused_latent = self.latent_fusion(concatenated_latent)  # [B, 40]
        
        if return_latents:
            return (fused_latent, state_action, action_state_recon, 
                   state_mu, state_logvar, state_z,
                   action_mu, action_logvar, action_z)
        
        return fused_latent, state_action, action_state_recon
    
    def compute_dual_loss(self, state, action, beta=1.0, recon_type="mse"):
        """
        Compute combined loss for both VAEs
        state: [B, 5, 84, 84]
        action: [B, 2]
        """
        # State->Action VAE loss
        state_action, state_mu, state_logvar, state_z = self.state_to_action_vae(state, return_latent=True)
        state_to_action_loss, state_recon_loss, state_kl = self.state_to_action_vae.loss(
            action, state_action, state_mu, state_logvar, beta=beta, recon_type=recon_type
        )
        
        # Action->State VAE loss
        action_state_recon, action_mu, action_logvar, action_z = self.action_to_state_vae(action, return_latent=True)
        action_to_state_loss, action_recon_loss, action_kl = self.action_to_state_vae.loss(
            state, action_state_recon, action_mu, action_logvar, beta=beta, recon_type=recon_type
        )
        
        # Combined loss
        total_loss = state_to_action_loss + action_to_state_loss
        
        return (total_loss, state_to_action_loss, action_to_state_loss,
                state_recon_loss, state_kl, action_recon_loss, action_kl)


class ImageVAE_CNN840Micro_LinearDec(nn.Module):
    """
    Encoder: CNN840_Micro -> [B,20,2] (flatten to 40)
    Latent:  z in R^{40}  (mu, logvar are both 40-dim)
    Decoder: Decoder40to5x84x84_LinearOnly expects [B,20,2]
    IO:
      x: [B,5,84,84]
      recon: [B,5,84,84]
    """
    def __init__(self, recon_activation=None):
        super().__init__()
        # 인코더: [B,5,84,84] -> [B,20,2]
        self.encoder = CNN840_Micro(out_steps=20, out_dim=2, base_ch=24, mid_ch=48, use_gn=True)
        # 잠복벡터 차원 (인코더 평탄화 크기와 동일하게 40으로 설정)
        self.latent_dim = 40

        # μ, logσ² 생성 헤드 (40 -> 40)
        self.fc_mu     = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # 디코더: [B,20,2] -> [B,5,84,84]
        # out_activation은 선택적으로 넣을 수 있음 (예: nn.Tanh(), nn.Sigmoid())
        self.decoder   = ActionDecoder(z_dim=self.latent_dim, out_dim=2, hidden=256)
    @staticmethod
    def reparameterize(mu, logvar):
        # logvar = log(σ^2) -> σ = exp(0.5 * logvar)
        logvar = torch.clamp(logvar, -10.0, 10.0)   # ★ 중요
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: [B,5,84,84]
        return: mu[ B,40 ], logvar[ B,40 ], z[ B,40 ]
        """
        feats = self.encoder(x)            # [B,20,2]
        h = feats.reshape(x.size(0), -1)   # [B,40]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        """
        z: [B,40]  -> reshape [B,20,2] -> decoder -> [B,5,84,84]
        """
        z_reshaped = z.view(z.size(0), 20, 2)
        recon = self.decoder(z_reshaped)
        return recon

    def forward(self, x, return_latent=False):
        """
        returns:
          recon, mu, logvar, (optional) z
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar,z

    def loss(self, x, recon, mu, logvar, beta=1.0, recon_type="mse"):
        """
        VAE loss = ReconLoss + beta * KL
        - recon_type: "mse" or "bce"
        - 입력 스케일:
            * mse: x가 [-1,1] 또는 [0,1] 어느 쪽이든 사용 가능
            * bce: x와 출력이 [0,1] 권장 (decoder에 Sigmoid 권장)
        """
        if recon_type.lower() == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        elif recon_type.lower() == "bce":
            # BCE는 입력/출력이 [0,1] 범위일 때 안정
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
        else:
            raise ValueError("recon_type must be 'mse' or 'bce'")

        # KL divergence: 0.5 * sum( μ^2 + σ^2 - logσ^2 - 1 )
        # 여기선 배치 평균으로 계산
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + beta * kl
        return total, recon_loss, kl

class FiLM(nn.Module):
    """
    Generate per-channel gamma/beta from z:[B,40] for a target channel size C.
    """
    def __init__(self, z_dim=40, c_out=16, hidden=256,scale=0.1):
        super().__init__()
        self.to_gamma = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, c_out))
        self.to_beta  = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, c_out))
        self.scale=scale

    def forward(self, zf, h):
        """
        zf:[B,40], h:[B,C,H,W]
        """
        zf=F.layer_norm(zf,(zf.size(1),))
        g = torch.tanh(self.to_gamma(zf))
        b = torch.tanh(self.to_beta(zf))
        gamma = (1.0 + self.scale * g).unsqueeze(-1).unsqueeze(-1)
        beta  = (self.scale * b).unsqueeze(-1).unsqueeze(-1)
        
        return gamma * h + beta

@dataclass
class PPOConfig:
    """Configuration class for PPO algorithm parameters"""
    ppo_eps: float = 0.2
    ppo_grad_descent_steps: int = 10
    gamma: float = 0.995
    lambda_gae: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    episodes_per_batch: int = 32
    train_epochs: int = 1000
    horizon: int = 300
    num_scenarios: int = 1000
    save_frequency: int = 100  # Save model every N steps
    save_final_model: bool = True  # Always save final model
    vae_lr:float=1e-4
    vae_beta:float=1.0
    vae_recon_type:str="mse"
    use_recon_for_policy:bool = True
    lambda_bc: float = 0.2
    lambda_entropy: float = 0.1
    lambda_z_value: float = 0.1
    detach_z_for_policy: bool = True  # start True; try False later


class Actor(nn.Module):
    """
    Actor network (Policy network) for PPO.
    Takes image observations and outputs continuous actions (throttle, steering).
    """
    
    def __init__(self, input_channels: int = 5,z_dim: int = 40):
        super().__init__()
        # Input: 84x84x5 (height, width, channels)
        # Output: 2 (throttle, steering)
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)  # 84x84x5 -> 20x20x16
       # self.bn1=nn.BatchNorm2d(16)
        self.bn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # 20x20x16 -> 9x9x32
        #self.bn2=nn.BatchNorm2d(32)
        self.bn2 = nn.GroupNorm(1, 32)
        self.fc1 = nn.Linear(9*9*32, 1024)  # 9x9x32 -> 1024
        self.lfc1=nn.Linear(1024,512)
        self.lfc2=nn.Linear(512,256)
        self.fc2 = nn.Linear(256+64, 2)  # 1024 -> 2
        self.z_flat=lambda z:z.view(z.size(0),-1)
        self.z_proj   = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 64), nn.ReLU(inplace=True))
        self.logstd_head = nn.Linear(256 + 64, 2)  # new
        self.film1=FiLM(z_dim=40,c_out=16,hidden=128)
        self.film2=FiLM(z_dim=40,c_out=32,hidden=128)
        
    def forward(self, x: torch.Tensor,z:torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Forward pass returning a multivariate normal distribution"""
        zf=self.z_flat(z)
        zp=self.z_proj(F.layer_norm(zf,(zf.size(1),)))
       # x = F.relu(self.conv1(x))
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(self.film1(zf,x))
        #x = F.relu(self.conv2(x))
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(self.film2(zf,x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.lfc1(x))
        x=F.relu(self.lfc2(x))
        h=torch.cat([x,zp],dim=1)
        mu = self.fc2(h)
        log_std=torch.clamp(self.logstd_head(h),-2.0,2.0)

        # Fixed standard deviation for simplicity
        #sigma = 0.05 * torch.ones_like(mu)
        sigma=log_std.exp()
        cov=torch.diag_embed(sigma**2)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)      # 가드
        cov = torch.nan_to_num(cov, nan=1e-6, posinf=1e6, neginf=1e-6)   # 가드
        return torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)

class Critic(nn.Module):
    def __init__(self, input_channels: int = 5, z_dim: int = 40):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 8, 4)
      #  self.bn1   = nn.BatchNorm2d(16)
        self.bn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
       # self.bn2   = nn.BatchNorm2d(32)
        self.bn2 = nn.GroupNorm(1, 32)
        self.z_proj = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(9*9*32 + 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        zf = z.view(z.size(0), -1)
        zp = self.z_proj(F.layer_norm(zf,(zf.size(1),)))
        x  = F.relu(self.bn1(self.conv1(x)))
        x  = F.relu(self.bn2(self.conv2(x)))
        x  = torch.flatten(x, 1)
        h  = torch.cat([x, zp], dim=1)
        return self.head(h).squeeze(-1)
class Critic2(nn.Module):
    """
    Critic network (Value network) for PPO.
    Takes image observations and outputs state values.
    """
    
    def __init__(self, input_channels: int = 5):
        super().__init__()
        # Input: 84x84x5 (height, width, channels)
        # Output: 1 (state value)
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)  # 84x84x5 -> 20x20x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # 20x20x16 -> 9x9x32
        self.fc1 = nn.Linear(9*9*32, 1024)  # 9x9x32 -> 256
        self.lfc1=nn.Linear(1024,512)
        self.lfc2=nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 1)  # 256 -> 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state values"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.lfc1(x))
        x=F.relu(self.lfc2(x))
        x = self.fc2(x)
        return torch.squeeze(x, dim=1)  # Remove batch dimension

class ZValue(nn.Module):
    def __init__(self, z_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, z): return self.net(z.view(z.size(0), -1)).squeeze(-1)

class NNPolicy:
    """Neural network policy wrapper"""

    def __init__(self, net: Actor, vae = None, use_recon_for_policy: bool = True):
        self.net = net
        self.vae = vae
        self.use_recon_for_policy = use_recon_for_policy

    def __call__(self, obs: npt.NDArray) -> Tuple[float, float]:
        """
        Sample action from policy given observation.
        If VAE is provided, build z (and optionally recon) online.
        """
        device = deviceof(self.net)
        obs_tensor = obs_batch_to_tensor([obs], device)  # [1,5,84,84]

        # z / recon 생성
        with torch.no_grad():
            if hasattr(self.vae, 'forward_with_fusion'):
                # DualVAE: Use forward_with_fusion for policy
                # For policy, we need to generate a dummy action to use forward_with_fusion
                dummy_action = torch.zeros(obs_tensor.size(0), 2, device=device)
                fused_latent, _, _ = self.vae.forward_with_fusion(obs_tensor, dummy_action)
                z = fused_latent
            elif hasattr(self.vae, 'state_to_action_vae'):
                # DualVAE: Use state->action VAE for policy
                action_pred, mu, logvar, z = self.vae.state_to_action_vae(obs_tensor, return_latent=True)
            else:
                # Single VAE: Original behavior
                action_pred, mu, logvar, z = self.vae(obs_tensor, return_latent=True)
        z = z.view(z.size(0), 20, 2)

        x_in = obs_tensor   # ✅ force image into Actor CNN
        dist = self.net(x_in, z)
        throttle, steering = dist.sample()[0]
        return throttle.item(), steering.item()

def deviceof(m: nn.Module) -> torch.device:
    """Get the device of the given module"""
    return next(m.parameters()).device


def obs_batch_to_tensor(obs: List[npt.NDArray[np.float32]], device: torch.device) -> torch.Tensor:
    """
    Convert observation batch to tensor and reshape from (B, H, W, C) to (B, C, H, W)
    """
    return torch.tensor(np.stack(obs), dtype=torch.float32, device=device).permute(0, 3, 1, 2)


def collect_trajectory(env: gym.Env, policy: Callable[[npt.NDArray], Tuple[float, float]]) -> Tuple[List[npt.NDArray], List[Tuple[float, float]], List[float]]:
    """
    Collect a trajectory from the environment using the given policy
    
    Returns:
        observations: List of observations
        actions: List of actions (throttle, steering)
        rewards: List of rewards
    """
    actor_mod = getattr(policy, "net", None)     # NNPolicy.net (Actor)
    vae_mod   = getattr(policy, "vae", None)     # NNPolicy.vae (VAE or None)
    was_actor_train = (actor_mod is not None) and actor_mod.training
    was_vae_train   = (vae_mod   is not None) and vae_mod.training
    

    if actor_mod is not None: actor_mod.eval()
    if vae_mod   is not None: vae_mod.eval()

    try:
        observations, actions, rewards = [], [], []
        obs, info = env.reset()

        while True:
            observations.append(obs)
            action = policy(obs)                     # actor/vae는 eval 모드
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        return observations, actions, rewards
    
    finally:
        # --- 여기서 원상복구 ---
        if was_actor_train and (actor_mod is not None):
            actor_mod.train()
        if was_vae_train and (vae_mod is not None):
            vae_mod.train()

    #return observations, actions, rewards


def rewards_to_go(trajectory_rewards: List[float], gamma: float) -> List[float]:
    """
    Compute the gamma discounted reward-to-go for each state in the trajectory
    """
    trajectory_len = len(trajectory_rewards)
    v_batch = np.zeros(trajectory_len)
    v_batch[-1] = trajectory_rewards[-1]

    # Use gamma to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + gamma * v_batch[t + 1]

    return list(v_batch)


def compute_advantage(
    critic: Critic,
    vae,  # Can be ImageVAE_CNN840Micro_LinearDec or DualVAEWithLatentFusion
    trajectory_observations: List[npt.NDArray[np.float32]],
    trajectory_rewards: List[float],
    gamma: float
) -> List[float]:
    """
    Compute advantage using GAE (Generalized Advantage Estimation)
    """
    trajectory_len = len(trajectory_rewards)
    assert len(trajectory_observations) == trajectory_len
    assert len(trajectory_rewards) == trajectory_len

    # Calculate the value of each state
    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        
        # Get latent from appropriate VAE
        if hasattr(vae, 'forward_with_fusion'):
            # DualVAE: Use forward_with_fusion
            # For advantage computation, we need dummy actions
            dummy_actions = torch.zeros(obs_tensor.size(0), 2, device=obs_tensor.device)
            fused_latent, _, _ = vae.forward_with_fusion(obs_tensor, dummy_actions)
            z = fused_latent
        elif hasattr(vae, 'state_to_action_vae'):
            # DualVAE: Use state->action VAE
            _, _, _, z = vae.state_to_action_vae(obs_tensor, return_latent=True)
        else:
            # Single VAE: Original behavior
            _, _, _, z = vae(obs_tensor, return_latent=True)
            
        z_reshape = z.view(z.size(0), 20, 2)
        obs_values = critic.forward(obs_tensor, z_reshape).detach().cpu().numpy()

    # Subtract the obs_value from the rewards-to-go
    trajectory_advantages = np.array(rewards_to_go(trajectory_rewards, gamma)) - obs_values
    return list(trajectory_advantages)


def compute_ppo_loss(
    pi_thetak_given_st: torch.distributions.MultivariateNormal,  # Old policy
    pi_theta_given_st: torch.distributions.MultivariateNormal,   # Current policy
    a_t: torch.Tensor,                                           # Actions taken
    A_pi_thetak_given_st_at: torch.Tensor,                       # Advantages
    config: PPOConfig
) -> torch.Tensor:
    """
    Compute PPO clipped loss
    """
    # Likelihood ratio (used to penalize divergence from the old policy)
    likelihood_ratio = torch.exp(pi_theta_given_st.log_prob(a_t) - pi_thetak_given_st.log_prob(a_t))

    # PPO clipped loss
    ppo_loss_per_example = -torch.minimum(
        likelihood_ratio * A_pi_thetak_given_st_at,
        torch.clip(likelihood_ratio, 1 - config.ppo_eps, 1 + config.ppo_eps) * A_pi_thetak_given_st_at,
    )

    return ppo_loss_per_example.mean()


def train_ppo(
    actor: Actor,
    critic: Critic,
    vae,  # Can be ImageVAE_CNN840Micro_LinearDec or DualVAEWithLatentFusion
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    vae_optimizier:torch.optim.Optimizer,
    z_value_model: ZValue,                          # NEW
    z_value_optimizer: torch.optim.Optimizer,       # NEW
    observation_batch: List[npt.NDArray[np.float32]],
    action_batch: List[Tuple[float, float]],
    advantage_batch: List[float],
    reward_to_go_batch: List[float],
    config: PPOConfig
) -> Tuple[List[float], List[float]]:
    """
    Train PPO for one batch of data
    """
    # Assert that models are on the same device
    assert deviceof(critic) == deviceof(actor)
    assert len(observation_batch) == len(action_batch)
    assert len(observation_batch) == len(advantage_batch)
    assert len(observation_batch) == len(reward_to_go_batch)

    device = deviceof(critic)

    # Convert data to tensors
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)
    true_value_batch_tensor = torch.tensor(reward_to_go_batch, dtype=torch.float32, device=device)
    chosen_action_tensor = torch.tensor(action_batch, device=device)
    advantage_batch_tensor = torch.tensor(advantage_batch, device=device)
    print(chosen_action_tensor.shape)
    # ------------------ VAE 학습 ------------------
    vae_optimizier.zero_grad()
    
    # Check if using DualVAE or single VAE
    if hasattr(vae, 'compute_dual_loss'):
        # DualVAE: Use dual loss for both state->action and action->state
        (vae_total, state_to_action_loss, action_to_state_loss,
         state_recon_loss, state_kl, action_recon_loss, action_kl) = vae.compute_dual_loss(
            observation_batch_tensor, chosen_action_tensor,
            beta=config.vae_beta, recon_type=config.vae_recon_type
        )
        # Get fused latent using forward_with_fusion
        print("Using forward_with_fusion for latent generation")
        fused_latent, _, _ = vae.forward_with_fusion(observation_batch_tensor, chosen_action_tensor)
        z = fused_latent  # Use the fused latent
        vae_recon = state_recon_loss + action_recon_loss
        vae_kl = state_kl + action_kl
    else:
        # Single VAE: Original behavior
        recon, mu, logvar, z = vae(observation_batch_tensor, return_latent=True)
        vae_total, vae_recon, vae_kl = vae.loss(
            chosen_action_tensor, recon, mu, logvar,
            beta=config.vae_beta, recon_type=config.vae_recon_type
        )
    
    vae_total.backward()
    vae_optimizier.step()
    # Train critic
    z_reshaped = z.detach().view(z.size(0), 20, 2)
    critic_optimizer.zero_grad()
    pred_value_batch_tensor = critic.forward(observation_batch_tensor,z_reshaped)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
    
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    critic_optimizer.step()

    # Train actor with PPO clipping
    with torch.no_grad():
        old_policy_action_probs = actor.forward(observation_batch_tensor,z_reshaped)
    adv_proc = advantage_batch_tensor
    adv_proc = (adv_proc - adv_proc.mean()) / (adv_proc.std() + 1e-8)
    adv_proc = adv_proc.clamp_(-3.0, 3.0)  # [-3, 3]로 클립
    
    pos_w = adv_proc.clamp(min=0.0)
    pos_w = pos_w / (pos_w.mean() + 1e-8)
    pos_w = pos_w.clamp_(max=3.0)
    actor_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
       # current_policy_action_probs = actor.forward(observation_batch_tensor,z_reshaped)
        dist=actor.forward(observation_batch_tensor,z_reshaped)
        actor_loss = compute_ppo_loss(
            old_policy_action_probs,
            #current_policy_action_probs,
            dist,
            chosen_action_tensor,
            adv_proc,
            config
        )
        mu_pred = dist.mean  # [B,2]
        #norm_adv = (advantage_batch_tensor - advantage_batch_tensor.mean()) / (advantage_batch_tensor.std() + 1e-8)
       # pos_w = norm_adv.clamp(min=0.0).detach()
        bc = F.mse_loss(mu_pred, chosen_action_tensor, reduction='none').mean(dim=1)
        bc_loss = (pos_w * bc).mean()

        # Entropy bonus
        ent = dist.entropy().mean()
        entropy_min=0.5
        entropy_floor_penalty=torch.relu(entropy_min-ent)

        # z-only value (optional, needs z_value module + optimizer)
        z_value_optimizer.zero_grad()
        v_z = z_value_model(z_reshaped.detach())
        z_value_loss = F.mse_loss(v_z, true_value_batch_tensor)
        z_value_loss.backward()
        z_value_optimizer.step()
        total_actor = actor_loss + config.lambda_bc*bc_loss - config.lambda_entropy*ent +0.01*entropy_floor_penalty+ config.lambda_z_value*z_value_loss.detach()
        total_actor.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        #actor_loss.backward()
        actor_optimizer.step()
        
        actor_losses.append(float(total_actor))

    return actor_losses, [float(critic_loss)] * config.ppo_grad_descent_steps,float(vae_total), float(vae_recon), float(vae_kl)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PPOTrainer:
    """PPO Trainer class for MetaDrive environment"""
    
    def __init__(self, config: PPOConfig = None, use_dual_vae: bool = True):
        self.config = config or PPOConfig()
       # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=torch.device("cpu")
        self.use_dual_vae = use_dual_vae
        
        # Initialize networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        
        # Choose VAE type
        if use_dual_vae:
            self.vae_guide = DualVAEWithLatentFusion(latent_dim=40, recon_activation=None).to(self.device)
            print("Using DualVAE with Latent Fusion")
        else:
            self.vae_guide = ImageVAE_CNN840Micro_LinearDec(recon_activation=None).to(self.device)
            print("Using Single State->Action VAE")
            
        self.z_value=ZValue(z_dim=40).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.vae_optimizer=torch.optim.Adam(self.vae_guide.parameters(),lr=self.config.vae_lr)
        
        self.z_value_optimizer=torch.optim.Adam(self.z_value.parameters(),lr=1e-4)
        # Initialize policy
        self.policy = NNPolicy(self.actor, vae=self.vae_guide, use_recon_for_policy=self.config.use_recon_for_policy)
        
        # Training statistics
        self.returns = []
        self.actor_losses = []
        self.critic_losses = []
        self.vae_total_losses=[]
        self.vae_recon_losses=[]
        self.vae_kl_losses=[]
        self.step = 0
        self.best_avg_return = float('-inf')  # Track best performance

    def create_env(self, render: bool = False) -> gym.Env:
        """Create MetaDrive environment"""
        return gym.make(
            "MetaDrive-topdown", 
            config={
                "use_render": render, 
                "horizon": self.config.horizon, 
                "num_scenarios": self.config.num_scenarios
            }
        )

    def train(self):
        """Main training loop"""
        env = self.create_env(render=False)
        
        print(f"Starting PPO training on {self.device}")
        print(f"Config: {self.config}")
        
        while self.step < self.config.train_epochs:
            # Collect batch of trajectories
            obs_batch = []
            act_batch = []
            rtg_batch = []
            adv_batch = []
            trajectory_returns = []

            for _ in range(self.config.episodes_per_batch):
                # Collect trajectory
                obs_traj, act_traj, rew_traj = collect_trajectory(env, self.policy)
                rtg_traj = rewards_to_go(rew_traj, self.config.gamma)
                adv_traj = compute_advantage(self.critic, self.vae_guide, obs_traj, rew_traj, self.config.gamma)

                # Update batch
                obs_batch.extend(obs_traj)
                act_batch.extend(act_traj)
                rtg_batch.extend(rtg_traj)
                adv_batch.extend(adv_traj)

                # Update trajectory returns
                trajectory_returns.append(sum(rew_traj))

            # Train on batch
            batch_actor_losses, batch_critic_losses ,vae_total, vae_recon, vae_kl= train_ppo(
                self.actor,
                self.critic,
                self.vae_guide,
                self.actor_optimizer,
                self.critic_optimizer,
                self.vae_optimizer,
                self.z_value,
                self.z_value_optimizer,
                obs_batch,
                act_batch,
                adv_batch,
                rtg_batch,
                self.config,
            )

            # Collect statistics
            self.returns.append(trajectory_returns)
            self.actor_losses.extend(batch_actor_losses)
            self.critic_losses.extend(batch_critic_losses)
            self.vae_total_losses.append(vae_total)
            self.vae_recon_losses.append(vae_recon)
            self.vae_kl_losses.append(vae_kl)

            # Print progress
            avg_return = np.mean(trajectory_returns)
            std_return = np.std(trajectory_returns)
            median_return = np.median(trajectory_returns)
            
            print(f"Step {self.step} | "
                  f"AvgRet: {avg_return:.3f} +/- {std_return:.3f} (Med {median_return:.3f}) | "
                  f"Actor: {self.actor_losses[-1]:.3f} | Critic: {batch_critic_losses[-1]:.3f} | "
                  f"VAE(total/recon/KL): {vae_total:.4f}/{vae_recon:.4f}/{vae_kl:.4f}")
            # Save model periodically
            if (self.step + 1) % self.config.save_frequency == 0:
                self.save_model(f"ppo_metadrive_model_step_{self.step + 1}.pth")
                print(f"💾 Model saved at step {self.step + 1}")
            
            # Save best model if performance improved
            if avg_return > self.best_avg_return:
                self.best_avg_return = avg_return
                self.save_model("ppo_metadrive_best_model.pth")
                print(f"🏆 New best model saved! Average return: {avg_return:.3f}")

            self.step += 1

        env.close()
        
        # Save final model if enabled
        if self.config.save_final_model:
            self.save_model("ppo_metadrive_model.pth")
            print("💾 Final model saved as 'ppo_metadrive_model.pth'")

    def evaluate(self, num_episodes: int = 5, render: bool = True) -> float:
        """Evaluate the trained policy"""
        env = self.create_env(render=render)
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, act, rew = collect_trajectory(env, self.policy)
            total_rewards.append(sum(rew))
        
        env.close()
        avg_reward = np.mean(total_rewards)
        print(f"Evaluation over {num_episodes} episodes: {avg_reward:.3f} +/- {np.std(total_rewards):.3f}")
        return avg_reward

    def plot_training_progress(self):
        """Plot training progress"""
        if not self.returns:
            print("No training data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns plot
        return_medians = [np.median(returns) for returns in self.returns]
        return_means = [np.mean(returns) for returns in self.returns]
        return_stds = [np.std(returns) for returns in self.returns]
        
        axes[0, 0].plot(return_means, label="Mean", color='blue')
        axes[0, 0].plot(return_medians, label="Median", color='red')
        axes[0, 0].fill_between(
            range(len(return_means)), 
            np.array(return_means) - np.array(return_stds), 
            np.array(return_means) + np.array(return_stds), 
            alpha=0.3, color='blue'
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Average Return")
        axes[0, 0].set_title("Training Returns")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Actor losses
        axes[0, 1].plot(self.actor_losses, label="Actor Loss", color='green')
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Actor Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Critic losses
        axes[1, 0].plot(self.critic_losses, label="Critic Loss", color='orange')
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Scatter plot of returns
        xs = []
        ys = []
        for t, rets in enumerate(self.returns):
            for ret in rets:
                xs.append(t)
                ys.append(ret)
        axes[1, 1].scatter(xs, ys, alpha=0.2, s=1)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Return")
        axes[1, 1].set_title("Return Distribution")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str):
        """Save trained model"""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'vae_state_dict': self.vae_guide.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'config': self.config,
            'returns': self.returns,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'vae_total_losses': self.vae_total_losses,
            'vae_recon_losses': self.vae_recon_losses,
            'vae_kl_losses': self.vae_kl_losses,
            'use_dual_vae': self.use_dual_vae,  # Save VAE type information
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path} (DualVAE: {self.use_dual_vae})")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load VAE type information
        saved_use_dual_vae = checkpoint.get('use_dual_vae', False)
        if saved_use_dual_vae != self.use_dual_vae:
            print(f"Warning: Model was saved with use_dual_vae={saved_use_dual_vae}, but current trainer has use_dual_vae={self.use_dual_vae}")
            print("Consider recreating trainer with correct VAE type")
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.vae_guide.load_state_dict(checkpoint['vae_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.returns = checkpoint.get('returns', [])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self.vae_total_losses = checkpoint.get('vae_total_losses', [])
        self.vae_recon_losses = checkpoint.get('vae_recon_losses', [])
        self.vae_kl_losses = checkpoint.get('vae_kl_losses', [])
        print(f"Model loaded from {path} (DualVAE: {saved_use_dual_vae})")


def main():
    """Main function to run PPO training"""
    # Create custom config
    config = PPOConfig(
        ppo_eps=0.2,
        ppo_grad_descent_steps=10,
        gamma=0.995,
        actor_lr=1e-4,
        critic_lr=1e-4,
        episodes_per_batch=32,
        train_epochs=300,
        horizon=300,
        num_scenarios=300,
        vae_lr=1e-4,
        vae_beta=1.0,
        vae_recon_type="mse",
        use_recon_for_policy=True,
    )
    
    # Create trainer with DualVAE (set use_dual_vae=True for new architecture, False for original)
    trainer = PPOTrainer(config, use_dual_vae=True)
    
    # Train
    trainer.train()
    
    # Evaluate
    trainer.evaluate(num_episodes=5, render=True)
    
    # Plot results
    trainer.plot_training_progress()
    
    # Ensure model is saved (backup save)
    try:
        trainer.save_model("ppo_metadrive_model.pth")
        print("✅ Model successfully saved as 'ppo_metadrive_model.pth'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        print("Trying to save with timestamp...")
        import time
        timestamp = int(time.time())
        trainer.save_model(f"ppo_metadrive_model_backup_{timestamp}.pth")
        print(f"✅ Model saved as backup: 'ppo_metadrive_model_backup_{timestamp}.pth'")


if __name__ == "__main__":
    main()
