"""
PPO + Diffuse-CLoC-style latent guidance (MetaDrive, 84x84x5)
- Rollout: guide latent z, feed guided (recon/z) to actor
- Train: reuse the SAME guided z (and recon if used) for actor/critic/old_policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Callable, Optional
import gymnasium as gym
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from metadrive.envs.top_down_env import TopDownMetaDrive

from guided_latent import GuidanceCfg, RollingLatent, CostComputer, LatentGuidance

# ----------------- Env register -----------------
gym.register(id="MetaDrive-topdown", entry_point=TopDownMetaDrive, kwargs=dict(config={}))
logging.getLogger("metadrive.envs.base_env").setLevel(logging.WARNING)

# ----------------- Small blocks -----------------
class DWConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return self.act(x)

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

class DWSep(nn.Module):
    def __init__(self, cin, cout, act=True, gn=True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(1, cout) if gn else nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.norm(x)
        return self.act(x)

# ----------------- Enc/Dec/VAE -----------------
class CNN840_Micro(nn.Module):
    def __init__(self, out_steps=20, out_dim=2, base_ch=24, mid_ch=48, use_gn=True):
        super().__init__()
        self.out_steps, self.out_dim = out_steps, out_dim
        self.stem = nn.Sequential(
            nn.Conv2d(5, base_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, base_ch) if use_gn else nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.stage = DWSep2(base_ch, mid_ch, stride=2, use_gn=use_gn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(mid_ch, out_steps * out_dim, 1, bias=True)
    def forward(self, x):
        x = self.stem(x)           # [B, base_ch, 42,42]
        x = self.stage(x)          # [B, mid_ch, 21,21]
        x = self.pool(x)           # [B, mid_ch, 1,1]
        x = self.head(x).flatten(1)   # [B,40]
        return x.view(-1, self.out_steps, self.out_dim)  # [B,20,2]

class Decoder40to5x84x84_LinearOnly(nn.Module):
    def __init__(self, hidden_dims=(512, 1024), out_activation=None, chunk_size=8192):
        super().__init__()
        self.in_dim = 40
        self.out_ch, self.H, self.W = 5, 84, 84
        self.out_dim = self.out_ch * self.H * self.W  # 35280
        self.out_activation = out_activation
        self.chunk_size = chunk_size
        layers = []
        prev = self.in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, self.out_dim)
    def forward(self, z):  # z:[B,20,2]
        B = z.size(0)
        x = z.view(B, -1)           # [B,40]
        h = self.mlp(x)             # [B,Hid]
        if self.chunk_size is None or self.chunk_size >= self.out_dim:
            y = self.fc_out(h)
        else:
            outs = []
            start = 0
            while start < self.out_dim:
                end = min(start + self.chunk_size, self.out_dim)
                W = self.fc_out.weight[start:end, :]
                b = self.fc_out.bias[start:end]
                outs.append(h @ W.t() + b)
                start = end
            y = torch.cat(outs, dim=1)
        if self.out_activation is not None:
            y = self.out_activation(y)
        return y.view(B, self.out_ch, self.H, self.W)

class ImageVAE_CNN840Micro_LinearDec(nn.Module):
    def __init__(self, recon_activation=None):
        super().__init__()
        self.encoder = CNN840_Micro(out_steps=20, out_dim=2, base_ch=24, mid_ch=48, use_gn=True)
        self.latent_dim = 40
        self.fc_mu     = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder = Decoder40to5x84x84_LinearOnly(hidden_dims=(512, 1024),
                                                     out_activation=recon_activation,
                                                     chunk_size=8192)
    @staticmethod
    def reparameterize(mu, logvar):
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def encode(self, x):
        feats = self.encoder(x)           # [B,20,2]
        h = feats.reshape(x.size(0), -1)  # [B,40]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
    def decode(self, z):                  # z:[B,40]
        z_reshaped = z.view(z.size(0), 20, 2)
        return self.decoder(z_reshaped)
    def forward(self, x, return_latent=False):
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        if return_latent: return recon, mu, logvar, z
        return recon, mu, logvar, z
    def loss(self, x, recon, mu, logvar, beta=1.0, recon_type="mse"):
        if recon_type.lower() == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        elif recon_type.lower() == "bce":
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
        else:
            raise ValueError("recon_type must be 'mse' or 'bce'")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + beta * kl
        return total, recon_loss, kl

# ----------------- Actor/Critic -----------------
class FiLM(nn.Module):
    def __init__(self, z_dim=40, c_out=16, hidden=256, scale=0.1):
        super().__init__()
        self.to_gamma = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, c_out))
        self.to_beta  = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, c_out))
        self.scale = scale
    def forward(self, zf, h):
        zf = F.layer_norm(zf, (zf.size(1),))
        g = torch.tanh(self.to_gamma(zf))
        b = torch.tanh(self.to_beta(zf))
        gamma = (1.0 + self.scale * g).unsqueeze(-1).unsqueeze(-1)
        beta  = (self.scale * b).unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta

@dataclass
class PPOConfig:
    ppo_eps: float = 0.2
    ppo_grad_descent_steps: int = 10
    gamma: float = 0.995
    lambda_gae: float = 0.95
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    episodes_per_batch: int = 32
    train_epochs: int = 1000
    horizon: int = 300
    num_scenarios: int = 1000
    save_frequency: int = 10
    save_final_model: bool = True
    vae_lr: float = 1e-4
    vae_beta: float = 1.0
    vae_recon_type: str = "mse"
    use_recon_for_policy: bool = True
    lambda_bc: float = 0.2
    lambda_entropy: float = 0.02
    lambda_z_value: float = 0.1
    detach_z_for_policy: bool = True

class Actor(nn.Module):
    def __init__(self, input_channels: int = 5, z_dim: int = 40):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.bn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.GroupNorm(1, 32)
        self.fc1 = nn.Linear(9*9*32, 1024)
        self.lfc1 = nn.Linear(1024, 512)
        self.lfc2 = nn.Linear(512, 256)

        self.z_proj = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 64), nn.ReLU(inplace=True))
        self.film1 = FiLM(z_dim=40, c_out=16, hidden=128)
        self.film2 = FiLM(z_dim=40, c_out=32, hidden=128)

        self.plan_head = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 8 * 2)
        )
        self.fc2 = nn.Linear(256 + 64, 2)
        self.logstd_head = nn.Linear(256 + 64, 2)

    def forward(self, x: torch.Tensor, z_seq: torch.Tensor) -> torch.distributions.MultivariateNormal:
        zf = z_seq.view(z_seq.size(0), -1)   # [B,40]
        zp = self.z_proj(F.layer_norm(zf, (zf.size(1),)))

        x = self.conv1(x); x = self.bn1(x); x = F.relu(self.film1(zf, x))
        x = self.conv2(x); x = self.bn2(x); x = F.relu(self.film2(zf, x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = F.relu(self.lfc1(x)); x = F.relu(self.lfc2(x))
        h = torch.cat([x, zp], dim=1)

        plan_seq = self.plan_head(h).view(h.size(0), 8, 2)
        mu = 0.5 * self.fc2(h) + 0.5 * plan_seq[:, 0, :]
        log_std = torch.clamp(self.logstd_head(h), -2.0, 2.0)
        sigma = log_std.exp()
        cov = torch.diag_embed(sigma**2)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cov = torch.nan_to_num(cov, nan=1e-6, posinf=1e6, neginf=1e-6)
        return torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)

class Critic(nn.Module):
    def __init__(self, input_channels: int = 5, z_dim: int = 40):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 8, 4)
        self.bn1   = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2   = nn.GroupNorm(1, 32)
        self.z_proj = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(9*9*32 + 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    def forward(self, x: torch.Tensor, z_seq: torch.Tensor) -> torch.Tensor:
        zf = z_seq.view(z_seq.size(0), -1)
        zp = self.z_proj(F.layer_norm(zf, (zf.size(1),)))
        x  = F.relu(self.bn1(self.conv1(x)))
        x  = F.relu(self.bn2(self.conv2(x)))
        x  = torch.flatten(x, 1)
        h  = torch.cat([x, zp], dim=1)
        return self.head(h).squeeze(-1)

class ZValue(nn.Module):
    def __init__(self, z_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, z): return self.net(z.view(z.size(0), -1)).squeeze(-1)

# ----------------- Helpers -----------------
def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device

def obs_batch_to_tensor(obs: List[npt.NDArray[np.float32]], device: torch.device) -> torch.Tensor:
    return torch.tensor(np.stack(obs), dtype=torch.float32, device=device).permute(0, 3, 1, 2)

# ----------------- Policy with guidance -----------------
class NNPolicy:
    def __init__(self, net: Actor, vae: Optional[ImageVAE_CNN840Micro_LinearDec]=None,
                 use_recon_for_policy: bool=True):
        self.net = net
        self.vae = vae
        self.use_recon_for_policy = use_recon_for_policy
        if self.vae is not None:
            self.guidance_cfg = GuidanceCfg(steps=6, step_size=0.12, prior_weight=0.2,
                                            use_recon_for_policy=use_recon_for_policy)
            self.roller = RollingLatent(noise_scale=0.03)
            self.latent_guidance = LatentGuidance(self.vae, self.guidance_cfg, self.roller)
            self._roller_inited = False

    def act_with_aux(self, obs: npt.NDArray):
        device = deviceof(self.net)
        obs_tensor = obs_batch_to_tensor([obs], device)
        if self.vae is None:
            z_seq = torch.zeros((1,20,2), device=device)
            dist  = self.net(obs_tensor, z_seq)
            a = dist.sample()[0]
            return (a[0].item(), a[1].item()), {'z': z_seq.detach().cpu().numpy(), 'recon': None}

        with torch.no_grad():
            recon0, mu, logvar, z0 = self.vae(obs_tensor, return_latent=True)
        if not getattr(self, "_roller_inited", False):
            self.roller.init(z0); self._roller_inited = True

        # 필요시 goal_px를 넣으세요. (예: torch.tensor([[y, x]], device=device))
        goal_px = None
        cost_fn = CostComputer(goal_px=goal_px, goal_weight=1.0, drivable_ch=0, obstacle_ch=None)

        z_guided, recon_guided = self.latent_guidance.guide(
            obs_tensor, mu, logvar, z0, cost_fn, return_recon=True
        )
        z_seq = z_guided.view(1,20,2)
        x_in  = recon_guided if self.use_recon_for_policy else obs_tensor
        dist  = self.net(x_in, z_seq)
        a = dist.sample()[0]
        return (a[0].item(), a[1].item()), {
            'z': z_seq.detach().cpu().numpy(),
            'recon': (recon_guided.detach().cpu().numpy() if self.use_recon_for_policy else None)
        }

# ----------------- Rollout -----------------
def collect_trajectory(env: gym.Env, policy: NNPolicy):
    """
    returns:
      observations: List[obs ndarray]
      actions:      List[(throttle, steer)]
      rewards:      List[float]
      guided_zs:    List[np.ndarray of shape [1,20,2]]
      guided_recs:  List[np.ndarray of shape [1,5,84,84]] or None
    """
    actor_mod = getattr(policy, "net", None)
    vae_mod   = getattr(policy, "vae", None)
    was_actor_train = (actor_mod is not None) and actor_mod.training
    was_vae_train   = (vae_mod   is not None) and vae_mod.training
    if actor_mod is not None: actor_mod.eval()
    if vae_mod   is not None: vae_mod.eval()

    try:
        observations, actions, rewards = [], [], []
        guided_zs, guided_recons = [], []
        obs, info = env.reset()
        while True:
            observations.append(obs)
            action, aux = policy.act_with_aux(obs)
            actions.append(action)
            guided_zs.append(aux['z'])
            guided_recons.append(aux['recon'])
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        return observations, actions, rewards, guided_zs, guided_recons
    finally:
        if was_actor_train and (actor_mod is not None): actor_mod.train()
        if was_vae_train   and (vae_mod   is not None): vae_mod.train()

# ----------------- Returns/Advantage -----------------
def rewards_to_go(trajectory_rewards: List[float], gamma: float) -> List[float]:
    T = len(trajectory_rewards)
    v = np.zeros(T, dtype=np.float32)
    v[-1] = trajectory_rewards[-1]
    for t in reversed(range(T-1)):
        v[t] = trajectory_rewards[t] + gamma * v[t+1]
    return list(v)

def compute_advantage(
    critic: Critic,
    trajectory_observations: List[npt.NDArray[np.float32]],
    trajectory_rewards: List[float],
    guided_z_traj: List[np.ndarray],
    gamma: float
) -> List[float]:
    """
    GAE 간소형: R2G - V(s; guided_z). 학습/롤아웃 분포 일치!
    """
    T = len(trajectory_rewards)
    assert len(trajectory_observations) == T
    assert len(guided_z_traj) == T

    device = deviceof(critic)
    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(trajectory_observations, device)
        z_seq = torch.tensor(np.concatenate(guided_z_traj, axis=0),
                             device=device, dtype=obs_tensor.dtype)  # [T,20,2]
        obs_values = critic(obs_tensor, z_seq).detach().cpu().numpy()

    adv = np.array(rewards_to_go(trajectory_rewards, gamma)) - obs_values
    return list(adv)

# ----------------- PPO loss -----------------
def compute_ppo_loss(
    pi_old: torch.distributions.MultivariateNormal,
    pi_new: torch.distributions.MultivariateNormal,
    a_t: torch.Tensor,
    A_t: torch.Tensor,
    eps: float
) -> torch.Tensor:
    ratio = torch.exp(pi_new.log_prob(a_t) - pi_old.log_prob(a_t))
    clipped = torch.clip(ratio, 1 - eps, 1 + eps)
    return -torch.minimum(ratio * A_t, clipped * A_t).mean()

# ----------------- Train one batch -----------------
def train_ppo(
    actor: Actor,
    critic: Critic,
    vae: ImageVAE_CNN840Micro_LinearDec,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    vae_optimizer: torch.optim.Optimizer,
    z_value_model: ZValue,
    z_value_optimizer: torch.optim.Optimizer,
    observation_batch: List[npt.NDArray[np.float32]],
    action_batch: List[Tuple[float, float]],
    advantage_batch: List[float],
    reward_to_go_batch: List[float],
    guided_z_batch: List[np.ndarray],
    guided_recon_batch: Optional[List[np.ndarray]],
    config: PPOConfig
):
    assert deviceof(critic) == deviceof(actor)
    device = deviceof(critic)

    # tensors
    obs_tensor = obs_batch_to_tensor(observation_batch, device)                # [B,5,84,84]
    actions = torch.tensor(action_batch, device=device, dtype=obs_tensor.dtype)# [B,2]
    A = torch.tensor(advantage_batch, device=device, dtype=obs_tensor.dtype)   # [B]
    R = torch.tensor(reward_to_go_batch, device=device, dtype=obs_tensor.dtype)# [B]
    z_seq = torch.tensor(np.concatenate(guided_z_batch, axis=0),
                         device=device, dtype=obs_tensor.dtype)                # [B,20,2]

    # x_in for actor (match rollout!)
    if config.use_recon_for_policy and guided_recon_batch is not None and guided_recon_batch[0] is not None:
        x_in = torch.tensor(np.concatenate(guided_recon_batch, axis=0),
                            device=device, dtype=obs_tensor.dtype)             # [B,5,84,84]
    else:
        x_in = obs_tensor

    # --------- VAE self-recon training (optional) ---------
    vae_optimizer.zero_grad()
    recon, mu, logvar, z = vae(obs_tensor, return_latent=True)
    vae_total, vae_recon, vae_kl = vae.loss(obs_tensor, recon, mu, logvar,
                                            beta=config.vae_beta,
                                            recon_type=config.vae_recon_type)
    vae_total.backward()
    vae_optimizer.step()

    # --------- Critic (on guided inputs!) ---------
    critic_optimizer.zero_grad()
    v_pred = critic(obs_tensor, z_seq)
    critic_loss = F.mse_loss(v_pred, R)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    critic_optimizer.step()

    # --------- Actor (PPO + BC + entropy + z-value) ---------
    with torch.no_grad():
        pi_old = actor(x_in, z_seq)  # snapshot on guided inputs

    # advantage proc
    A = (A - A.mean()) / (A.std() + 1e-8)
    A = A.clamp_(-3.0, 3.0)
    pos_w = A.clamp(min=0.0)
    pos_w = (pos_w / (pos_w.mean() + 1e-8)).clamp_(max=3.0)

    actor_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
        pi_new = actor(x_in, z_seq)

        ppo_loss = compute_ppo_loss(pi_old, pi_new, actions, A, config.ppo_eps)

        # AWR-style BC (weighted MSE to taken actions)
        mu_pred = pi_new.mean
        bc = F.mse_loss(mu_pred, actions, reduction='none').mean(dim=1)
        bc_loss = (pos_w * bc).mean()

        # entropy bonus (+ floor penalty to avoid collapse)
        ent = pi_new.entropy().mean()
        entropy_min = 0.5
        entropy_floor_penalty = torch.relu(entropy_min - ent)

        # z-only value (optional)
        z_value_optimizer.zero_grad()
        v_z = z_value_model(z_seq.detach())
        z_value_loss = F.mse_loss(v_z, R)
        z_value_loss.backward()
        z_value_optimizer.step()

        total_actor = ppo_loss + config.lambda_bc * bc_loss \
                      - config.lambda_entropy * ent + 0.01 * entropy_floor_penalty \
                      + config.lambda_z_value * z_value_loss.detach()
        total_actor.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()
        actor_losses.append(float(total_actor))

    return actor_losses, [float(critic_loss)] * config.ppo_grad_descent_steps, \
           float(vae_total), float(vae_recon), float(vae_kl)

# ----------------- Trainer -----------------
class PPOTrainer:
    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.device = torch.device("cpu")
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.vae_guide = ImageVAE_CNN840Micro_LinearDec(recon_activation=None).to(self.device)
        self.z_value = ZValue(z_dim=40).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.vae_optimizer = torch.optim.Adam(self.vae_guide.parameters(), lr=self.config.vae_lr)
        self.z_value_optimizer = torch.optim.Adam(self.z_value.parameters(), lr=1e-4)

        self.policy = NNPolicy(self.actor, vae=self.vae_guide,
                               use_recon_for_policy=self.config.use_recon_for_policy)

        self.returns = []
        self.actor_losses = []
        self.critic_losses = []
        self.vae_total_losses = []
        self.vae_recon_losses = []
        self.vae_kl_losses = []
        self.step = 0
        self.best_avg_return = float('-inf')

    def create_env(self, render: bool = False) -> gym.Env:
        return gym.make("MetaDrive-topdown",
                        config={"use_render": render,
                                "horizon": self.config.horizon,
                                "num_scenarios": self.config.num_scenarios})

    def train(self):
        env = self.create_env(render=False)
        print(f"Starting PPO training on {self.device}")
        print(f"Config: {self.config}")

        while self.step < self.config.train_epochs:
            obs_batch, act_batch, rtg_batch, adv_batch = [], [], [], []
            z_batch, recon_batch = [], []
            trajectory_returns = []

            for _ in range(self.config.episodes_per_batch):
                obs_traj, act_traj, rew_traj, z_traj, rec_traj = collect_trajectory(env, self.policy)
                rtg_traj = rewards_to_go(rew_traj, self.config.gamma)
                adv_traj = compute_advantage(self.critic, obs_traj, rew_traj, z_traj, self.config.gamma)

                obs_batch.extend(obs_traj)
                act_batch.extend(act_traj)
                rtg_batch.extend(rtg_traj)
                adv_batch.extend(adv_traj)
                z_batch.extend(z_traj)
                recon_batch.extend(rec_traj)

                trajectory_returns.append(sum(rew_traj))

            a_losses, c_losses, vae_total, vae_recon, vae_kl = train_ppo(
                self.actor, self.critic, self.vae_guide,
                self.actor_optimizer, self.critic_optimizer, self.vae_optimizer,
                self.z_value, self.z_value_optimizer,
                obs_batch, act_batch, adv_batch, rtg_batch,
                z_batch, recon_batch, self.config
            )

            self.returns.append(trajectory_returns)
            self.actor_losses.extend(a_losses)
            self.critic_losses.extend(c_losses)
            self.vae_total_losses.append(vae_total)
            self.vae_recon_losses.append(vae_recon)
            self.vae_kl_losses.append(vae_kl)

            avg_return = np.mean(trajectory_returns)
            std_return = np.std(trajectory_returns)
            median_return = np.median(trajectory_returns)

            print(f"Step {self.step} | "
                  f"AvgRet: {avg_return:.3f} +/- {std_return:.3f} (Med {median_return:.3f}) | "
                  f"Actor: {self.actor_losses[-1]:.3f} | Critic: {c_losses[-1]:.3f} | "
                  f"VAE(t/r/KL): {vae_total:.4f}/{vae_recon:.4f}/{vae_kl:.4f}")

            if (self.step + 1) % self.config.save_frequency == 0:
                self.save_model(f"ppo_metadrive_model_step_{self.step + 1}.pth")
                print(f"💾 Model saved at step {self.step + 1}")

            if avg_return > self.best_avg_return:
                self.best_avg_return = avg_return
                self.save_model("ppo_metadrive_best_model.pth")
                print(f"🏆 New best model saved! Average return: {avg_return:.3f}")

            self.step += 1

        env.close()
        if self.config.save_final_model:
            self.save_model("ppo_metadrive_model.pth")
            print("💾 Final model saved as 'ppo_metadrive_model.pth'")

    def evaluate(self, num_episodes: int = 5, render: bool = True) -> float:
        env = self.create_env(render=render)
        totals = []
        for _ in range(num_episodes):
            obs, act, rew, _, _ = collect_trajectory(env, self.policy)
            totals.append(sum(rew))
        env.close()
        avg = float(np.mean(totals))
        print(f"Evaluation over {num_episodes} episodes: {avg:.3f} +/- {np.std(totals):.3f}")
        return avg

    def plot_training_progress(self):
        if not self.returns:
            print("No training data to plot"); return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        return_medians = [np.median(returns) for returns in self.returns]
        return_means = [np.mean(returns) for returns in self.returns]
        return_stds  = [np.std(returns) for returns in self.returns]
        axes[0,0].plot(return_means, label="Mean"); axes[0,0].plot(return_medians, label="Median")
        axes[0,0].fill_between(range(len(return_means)),
                               np.array(return_means)-np.array(return_stds),
                               np.array(return_means)+np.array(return_stds),
                               alpha=0.3)
        axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Average Return")
        axes[0,0].set_title("Training Returns"); axes[0,0].legend(); axes[0,0].grid(True)

        axes[0,1].plot(self.actor_losses, label="Actor Loss"); axes[0,1].legend(); axes[0,1].grid(True)
        axes[1,0].plot(self.critic_losses, label="Critic Loss"); axes[1,0].legend(); axes[1,0].grid(True)

        xs, ys = [], []
        for t, rets in enumerate(self.returns):
            for r in rets: xs.append(t); ys.append(r)
        axes[1,1].scatter(xs, ys, alpha=0.2, s=1)
        axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Return")
        axes[1,1].set_title("Return Distribution"); axes[1,1].grid(True)
        plt.tight_layout(); plt.show()

    def save_model(self, path: str):
        torch.save({
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
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.vae_guide.load_state_dict(ckpt['vae_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
        self.vae_optimizer.load_state_dict(ckpt['vae_optimizer_state_dict'])
        self.returns = ckpt.get('returns', [])
        self.actor_losses = ckpt.get('actor_losses', [])
        self.critic_losses = ckpt.get('critic_losses', [])
        self.vae_total_losses = ckpt.get('vae_total_losses', [])
        self.vae_recon_losses = ckpt.get('vae_recon_losses', [])
        self.vae_kl_losses = ckpt.get('vae_kl_losses', [])
        print(f"Model loaded from {path}")

# ----------------- Main -----------------
def main():
    config = PPOConfig(
        ppo_eps=0.2, ppo_grad_descent_steps=10, gamma=0.995,
        actor_lr=1e-4, critic_lr=1e-4,
        episodes_per_batch=8,     # NOTE: 샘플 테스트 시 8로 낮춤 (GPU/시간에 맞게 조정)
        train_epochs=1000,
        horizon=300, num_scenarios=300,
        vae_lr=1e-4, vae_beta=1.0, vae_recon_type="mse",
        use_recon_for_policy=True,
    )
    trainer = PPOTrainer(config)
    trainer.train()
    trainer.evaluate(num_episodes=5, render=True)
    trainer.plot_training_progress()
    try:
        trainer.save_model("ppo_metadrive_model.pth")
        print("✅ Model successfully saved as 'ppo_metadrive_model.pth'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        import time
        ts = int(time.time())
        trainer.save_model(f"ppo_metadrive_model_backup_{ts}.pth")
        print(f"✅ Model saved as backup: 'ppo_metadrive_model_backup_{ts}.pth'")

if __name__ == "__main__":
    main()
