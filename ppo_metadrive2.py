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
import vae_motion
from vae_motion import VAE
from pathlib import Path

def _resolve_same_dir_path(rel_name: str) -> Path:
    """현재 .py 파일과 같은 폴더 기준으로 상대 경로를 절대 경로로 변환.
       __file__이 없는 노트북/REPL 환경에선 CWD를 사용."""
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    return (base / rel_name).expanduser().resolve()
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
@dataclass
class PPOConfig:
    """Configuration class for PPO algorithm parameters"""
    ppo_eps: float = 0.2
    ppo_grad_descent_steps: int = 10
    gamma: float = 0.995
    lambda_gae: float = 0.95
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    episodes_per_batch: int = 32
    train_epochs: int = 1000
    horizon: int = 300
    num_scenarios: int = 100
    save_frequency: int = 10  # Save model every N steps
    save_final_model: bool = True  # Always save final model
    vae_path:Optional[str]="vae_argo_best.pt"
    use_feature:bool = True
    m2m_weight:float =1.0
    m2m_lr:float =1e-4


class Actor(nn.Module):
    """
    Actor network (Policy network) for PPO.
    Takes image observations and outputs continuous actions (throttle, steering).
    """
    
    def __init__(self, input_channels: int = 5):
        super().__init__()
        # Input: 84x84x5 (height, width, channels)
        # Output: 2 (throttle, steering)
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)  # 84x84x5 -> 20x20x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # 20x20x16 -> 9x9x32
        self.fc1 = nn.Linear(9*9*32, 512)  # 9x9x32 -> 1024
        self.lfc1=nn.Linear(512,256)
        #self.lfc2=nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 2)  # 1024 -> 2

    def forward(self, x: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Forward pass returning a multivariate normal distribution"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.lfc1(x))
        #x=F.relu(self.lfc2(x))
        mu = self.fc2(x)
        
        # Fixed standard deviation for simplicity
        sigma = 0.05 * torch.ones_like(mu)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))


class Critic(nn.Module):
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
        self.fc1 = nn.Linear(9*9*32, 512)  # 9x9x32 -> 256
        self.lfc1=nn.Linear(512,256)
        #elf.lfc2=nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 1)  # 256 -> 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state values"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.lfc1(x))
        #x=F.relu(self.lfc2(x))
        x = self.fc2(x)
        return torch.squeeze(x, dim=1)  # Remove batch dimension


class NNPolicy:
    """Neural network policy wrapper"""
    
    def __init__(self, net: Actor):
        self.net = net

    def __call__(self, obs: npt.NDArray) -> Tuple[float, float]:
        """Sample action from policy given observation"""
        obs_tensor = obs_batch_to_tensor([obs], deviceof(self.net))
        with torch.no_grad():
            throttle, steering = self.net(obs_tensor).sample()[0]
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
    observations = []
    actions = []
    rewards = []
    obs, info = env.reset()
    
    while True:
        observations.append(obs)
        action = policy(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    return observations, actions, rewards


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
        obs_values = critic.forward(obs_tensor).detach().cpu().numpy()

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
class FusionConv(nn.Module):
    def __init__(self,in_ch=10,out_ch=5):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,16,kernel_size=1,bias=False),
            nn.BatchNorm2d(16),nn.ReLU(inplace=True),
            nn.Conv2d(16,out_ch,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)
# ---- 1) Patchify / Unpatchify ----
class Patchify(nn.Module):
    def __init__(self, in_ch=5, img_h=84, img_w=84, patch=6):
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        self.in_ch, self.H, self.W, self.P = in_ch, img_h, img_w, patch
        self.nh = img_h // patch
        self.nw = img_w // patch
        self.num_tokens = self.nh * self.nw     # = 14*14 = 196
        self.vec_dim = in_ch * patch * patch    # = 5*6*6 = 180

    def forward(self, x):              # x: [B, C=5, 84, 84]
        B, C, H, W = x.shape
        x = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)  # [B,C,nh,nw,P,P]
        x = x.permute(0,2,3,1,4,5).contiguous().view(B, self.num_tokens, self.vec_dim)  # [B,196,180]
        return x

class Unpatchify(nn.Module):
    def __init__(self, out_ch=5, img_h=84, img_w=84, patch=6):
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        self.C, self.H, self.W, self.P = out_ch, img_h, img_w, patch
        self.nh = img_h // patch
        self.nw = img_w // patch
        self.vec_dim = out_ch * patch * patch

    def forward(self, tokens):         # [B, 196, 180]
        B, N, V = tokens.shape
        assert N == self.nh * self.nw and V == self.vec_dim
        x = tokens.view(B, self.nh, self.nw, self.C, self.P, self.P)  # [B,nh,nw,C,P,P]
        x = x.permute(0,3,1,4,2,5).contiguous().view(B, self.C, self.H, self.W)  # [B,C,H,W]
        return x

# ---- 2) Positional Encoding (learnable or sinusoidal) ----
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)      # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)          # not a parameter

    def forward(self, x):                       # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

# ---- 3) 전체 Wrapper: src/tgt -> Transformer -> 복원 ----
class FusionTransformer(nn.Module):
    """
    src, tgt: [B, 5, 84, 84]
    출력: tgt와 같은 해상도의 예측(또는 디코더 마지막 토큰 예측 등)
    """
    def __init__(self, in_ch=5, img_h=84, img_w=84, patch=6,
                 d_model=1, nhead=8, num_enc=1, num_dec=1, dim_ff=256, dropout=0.1):
        super().__init__()
        self.patch = Patchify(in_ch, img_h, img_w, patch)
        self.unpatch = Unpatchify(in_ch, img_h, img_w, patch)
        self.vec_dim = in_ch * patch * patch       # 180
        self.num_tokens = (img_h // patch) * (img_w // patch)  # 196

        # 180 -> d_model
        self.src_proj = nn.Linear(self.vec_dim, d_model)
        self.tgt_proj = nn.Linear(self.vec_dim, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=self.num_tokens)

        self.tf = nn.Transformer(
            d_model=d_model, nhead=1,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )

        # d_model -> 180 (패치벡터 복원)
        self.out_head = nn.Linear(d_model, self.vec_dim)

    def _causal_mask(self, L):
        # 표준 Transformer 디코더용 causal mask (자동완성/미래가림이 필요할 때)
        mask = torch.full((L, L), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, src_img, tgt_img, use_causal_tgt=False):
        """
        src_img, tgt_img: [B, 5, 84, 84]
        반환: [B, 5, 84, 84]
        """
        B = src_img.size(0)

        # 1) patchify -> [B,196,180]
        src_tokens = self.patch(src_img)
        tgt_tokens = self.patch(tgt_img)

        # 2) 180 -> d_model & pos enc
        src = self.pos_enc(self.src_proj(src_tokens))   # [B,196,D]
        tgt = self.pos_enc(self.tgt_proj(tgt_tokens))   # [B,196,D]

        # 3) optional mask (디코더는 미래 가림 가능)
        tgt_mask = None
        if use_causal_tgt:
            tgt_mask = self._causal_mask(tgt.size(1)).to(tgt.device)  # [L,L]

        # 4) Transformer
        out = self.tf(src=src, tgt=tgt, tgt_mask=tgt_mask)  # [B,196,D]

        # 5) D -> 180 -> unpatchify
        out_tokens = self.out_head(out)         # [B,196,180]
        out_img = self.unpatch(out_tokens)      # [B,5,84,84]
        return out_img
def train_ppo(
    actor: Actor,
    critic: Critic,
    mid1,
    mid2,
    vae_model,
    fuser:FusionTransformer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    m2m_optimizer:torch.optim.Optimizer,
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
    #print(observation_batch_tensor.shape)
    fused_obs=observation_batch_tensor
    z=mid1(observation_batch_tensor)
    feature_img_base=mid2(z)
    loss_M2M=F.mse_loss(feature_img_base,observation_batch_tensor)
    self_m2m_loss=loss_M2M*config.m2m_weight
    m2m_optimizer.zero_grad()
    self_m2m_loss.backward()
    m2m_optimizer.step()
    if config.use_feature and (vae_model is not None):
        with torch.no_grad():
          #  z=mid1(observation_batch_tensor)
            try:
                #print("vae_사용중")
                out,_,_,mu,logvar = vae_model(z)
            except Exception:
                out = z 
            if isinstance(out,(tuple,list)):
                out=out[0]
            if out.dim() == 3:
                #print("3번 작동중")
                feature_img=mid2(out)
           #     feature_img_base=mid2(z)
            elif out.dim()==4 and out.shape[1]==5:
                feature_img = out
            else:
                feature_img = mid2(z)
            
            #loss_M2M=F.mse_loss(feature_img_base,observation_batch_tensor)
            #self_m2m_loss=loss_M2M*config.m2m_weight
            #m2m_optimizer.zero_grad()
            #self_m2m_loss.backward()
            #m2m_optimizer.step()
            # print(feature_img.shape)
            cat = torch.cat([observation_batch_tensor,feature_img],dim=1)
            fused_obs=fuser(observation_batch_tensor,feature_img)
            fused_obs=fused_obs.detach()

        #observation_batch_tensor=mid2(mid1(observation_batch_tensor))
    # Train critic
    critic_optimizer.zero_grad()
    #pred_value_batch_tensor = critic.forward(observation_batch_tensor)
    pred_value_batch_tensor = critic.forward(observation_batch_tensor)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
    critic_loss.backward()
    critic_optimizer.step()

    # Train actor with PPO clipping
    with torch.no_grad():
        old_policy_action_probs=actor.forward(fused_obs)
        #old_policy_action_probs = actor.forward(observation_batch_tensor)

    actor_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
        #current_policy_action_probs = actor.forward(observation_batch_tensor)
        current_policy_action_probs = actor.forward(fused_obs)
        actor_loss = compute_ppo_loss(
            old_policy_action_probs,
            current_policy_action_probs,
            chosen_action_tensor,
            advantage_batch_tensor,
            config
        )
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(float(actor_loss))

    return actor_losses, [float(critic_loss)] * config.ppo_grad_descent_steps


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PPOTrainer:
    """PPO Trainer class for MetaDrive environment"""
    
    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # Initialize networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.M2A=CNN840_Micro().to(self.device)
        self.A2M=Decoder40to5x84x84_LinearOnly().to(self.device)
        self.model:Optional[nn.Module]=None

        if self.config.vae_path:
            try:
                ckpt_name=self.config.vae_path
                if isinstance(ckpt_name,(tuple,list)):
                    ckpt_name=ckpt_name[0]
                ckpt_path=_resolve_same_dir_path(ckpt_name)
            
                obj=torch.load(ckpt_path,map_location=self.device)
                #print(obj)
               # self.model=obj.to(self.device)
               # self.model.eval()
                if isinstance(obj,nn.Module):
                    self.model=obj.to(self.device)
                    self.model.eval()
                    print(f"✅ Loaded VAE module from: {ckpt_path}")
                #self.model=torch.load(ckpt_path,map_location=self.device)
                elif isinstance(obj,dict) and isinstance(obj.get("model"), nn.Module):
                    if isinstance(obj.get("model"),nn.Module):
                        print("vae작동중")
                        self.model=obj["model"].to(self.device)
                        self.model.eval()
                        print(self.model)
                        print(f"✅ Loaded VAE module from dict['model']: {ckpt_path}")
                    elif "state_dict" in obj:
                        print("⚠️ Checkpoint has a 'state_dict' but no model class to load into. "
                      "Skipping external VAE model, but you can use (B) or (C) below.")
                        self.model=None
                else:
                    print("⚠️ Unknown dict format; external VAE disabled.")
                    self.model=None
            except Exception as e:
                print(f"❌ Could not load VAE from {self.config.vae_path}: {e}")
                self.model=None


            




        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        self.m2m_optimizer = torch.optim.Adam(
            list(self.M2A.parameters())+list(self.A2M.parameters()),
            lr=self.config.m2m_lr
        )
        # Initialize policy
        self.policy = NNPolicy(self.actor)
        self.fuser=FusionTransformer().to(self.device)
        # Training statistics
        self.returns = []
        self.actor_losses = []
        self.critic_losses = []
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
                adv_traj = compute_advantage(self.critic, obs_traj, rew_traj, self.config.gamma)

                # Update batch
                obs_batch.extend(obs_traj)
                act_batch.extend(act_traj)
                rtg_batch.extend(rtg_traj)
                adv_batch.extend(adv_traj)

                # Update trajectory returns
                trajectory_returns.append(sum(rew_traj))

            # Train on batch
           # print(obs_batch.shape)
            batch_actor_losses, batch_critic_losses = train_ppo(
                self.actor,
                self.critic,
                self.M2A,
                self.A2M,
                self.model,
                self.fuser,
                self.actor_optimizer,
                self.critic_optimizer,
                self.m2m_optimizer,
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

            # Print progress
            avg_return = np.mean(trajectory_returns)
            std_return = np.std(trajectory_returns)
            median_return = np.median(trajectory_returns)
            
            print(f"Step {self.step}, Avg. Returns: {avg_return:.3f} +/- {std_return:.3f}, "
                  f"Median: {median_return:.3f}, Actor Loss: {self.actor_losses[-1]:.3f}, "
                  f"Critic Loss: {batch_critic_losses[-1]:.3f}")

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
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'returns': self.returns,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.returns = checkpoint.get('returns', [])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        print(f"Model loaded from {path}")


def main():
    """Main function to run PPO training"""
    # Create custom config
    config = PPOConfig(
        ppo_eps=0.2,
        ppo_grad_descent_steps=10,
        gamma=0.995,
        actor_lr=3e-4,
        critic_lr=1e-4,
        episodes_per_batch=32,
        train_epochs=500,
        horizon=300,
        num_scenarios=300
    )
    
    # Create trainer
    trainer = PPOTrainer(config)
    
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
