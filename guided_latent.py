# ===== guided_latent.py =====
import torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class GuidanceCfg:
    steps: int = 8               # guidance iterations per decision
    step_size: float = 0.15      # latent gradient step
    prior_weight: float = 0.25   # keep z near posterior (mu,logvar)
    smooth_weight: float = 0.0   # optional TV/smoothness on recon
    clamp_norm: Optional[float] = 5.0  # grad norm clip
    use_recon_for_policy: bool = True

class RollingLatent:
    """Warm-start z with small noise injection (rolling)."""
    def __init__(self, noise_scale=0.05):
        self.prev_z: Optional[torch.Tensor] = None
        self.noise_scale = noise_scale

    def init(self, z0: torch.Tensor):
        self.prev_z = z0.detach()

    def warm_start(self, mu: torch.Tensor, z0: torch.Tensor):
        if self.prev_z is None or self.prev_z.shape != z0.shape:
            self.prev_z = z0.detach()
        # convex mix toward current posterior mean + small noise
        z = 0.7 * self.prev_z + 0.3 * mu.detach()
        z = z + self.noise_scale * torch.randn_like(z)
        return z.detach()

    def update(self, z_new: torch.Tensor):
        self.prev_z = z_new.detach()

class CostComputer:
    """
    Pluggable costs computed on recon images (B,5,84,84) and/or actions.
    You can fill in any combination that matches your obs channels.
    """
    def __init__(self, goal_px: Optional[torch.Tensor]=None, goal_weight=1.0,
                 drivable_ch: int=0, obstacle_ch: Optional[int]=None):
        self.goal_px = goal_px          # (B,2) pixel coords (y,x) in [0,83]
        self.goal_weight = goal_weight
        self.drivable_ch = drivable_ch
        self.obstacle_ch = obstacle_ch

    def _goal_heatmap(self, B, H=84, W=84, sigma=5.0, device='cpu'):
        if self.goal_px is None: return None
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        yy = yy[None].expand(B,-1,-1); xx = xx[None].expand(B,-1,-1)
        gy = self.goal_px[:,0].view(B,1,1); gx = self.goal_px[:,1].view(B,1,1)
        g = torch.exp(-((yy-gy)**2 + (xx-gx)**2)/(2*sigma**2))  # [B,H,W]
        return g

    def cost(self, recon: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        recon: [B,5,84,84] (decoder output from latent)
        obs:   [B,5,84,84] (current obs, if you want mask references)
        returns scalar cost (averaged over batch)
        """
        B, C, H, W = recon.shape
        total = recon.sum()*0.0

        # (1) Goal attraction in image space (align channel 0/route to Gaussian blob)
        ghm = self._goal_heatmap(B, H, W, device=recon.device)
        if ghm is not None:
            # Encourage recon "route/ego-forward" intensity to match goal heatmap.
            # If your route channel is not 0, change it below.
            route_ch = 0
            pred = torch.sigmoid(recon[:, route_ch])     # [B,H,W]
            total = total + self.goal_weight * F.binary_cross_entropy(pred, ghm)

        # (2) Drivable area: penalize intensity outside drivable mask (taken from obs)
        drive_mask = torch.sigmoid(obs[:, self.drivable_ch])  # [B,H,W] in [0,1]
        # push recon channels to be low outside drivable
        outside = (1.0 - drive_mask)
        total = total + 0.05 * (outside * (recon**2).sum(dim=1)).mean()

        # (3) Obstacle penetration: if you have an obstacle channel in obs
        if self.obstacle_ch is not None:
            obst = torch.sigmoid(obs[:, self.obstacle_ch])  # [B,H,W]
            # discourage bright recon where obstacles are
            total = total + 0.05 * (obst * (recon**2).sum(dim=1)).mean()

        return total

class LatentGuidance:
    """
    Do a few gradient steps on z to reduce task cost C(recon) + prior term.
    This is 'classifier-guidance-like' in the VAE latent.
    """
    def __init__(self, vae, cfg: GuidanceCfg, roller: RollingLatent):
        self.vae = vae
        self.cfg = cfg
        self.roller = roller

    def guide(self, obs_tensor: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              z0: torch.Tensor,
              cost_fn: CostComputer,
              return_recon: bool = True):
        z = self.roller.warm_start(mu, z0)  # rolling init
        for _ in range(self.cfg.steps):
            z = z.detach().requires_grad_(True)
            recon = self.vae.decode(z)  # decoder expects [B,40] -> [B,5,84,84]
            cost = cost_fn.cost(recon, obs_tensor)
            # prior: keep z near posterior (like KL's pull)
            prior = ((z - mu) ** 2) * torch.exp(-logvar)   # ~ (z-mu)^2 / var
            prior = prior.mean()
            loss = cost + self.cfg.prior_weight * prior

            if self.cfg.smooth_weight > 0:
                # simple TV on recon to reduce noise (optional)
                tv = (recon[:, :, 1:, :] - recon[:, :, :-1, :]).abs().mean() + \
                     (recon[:, :, :, 1:] - recon[:, :, :, :-1]).abs().mean()
                loss = loss + self.cfg.smooth_weight * tv

            g = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            if self.cfg.clamp_norm is not None:
                g = torch.nan_to_num(g)
                gn = torch.norm(g, dim=1, keepdim=True) + 1e-6
                g = torch.where(gn > self.cfg.clamp_norm, g * (self.cfg.clamp_norm/gn), g)
            z = z - self.cfg.step_size * g

        self.roller.update(z)
        if return_recon or self.cfg.use_recon_for_policy:
            recon = self.vae.decode(z)
            return z.detach(), recon.detach()
        return z.detach(), None
