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
    num_scenarios: int = 1000
    save_frequency: int = 10  # Save model every N steps
    save_final_model: bool = True  # Always save final model


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
        self.fc1 = nn.Linear(9*9*32, 1024)  # 9x9x32 -> 1024
        self.lfc1=nn.Linear(1024,512)
        self.lfc2=nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 2)  # 1024 -> 2

    def forward(self, x: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Forward pass returning a multivariate normal distribution"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.lfc1(x))
        x=F.relu(self.lfc2(x))
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


def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
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

    # Train critic
    critic_optimizer.zero_grad()
    pred_value_batch_tensor = critic.forward(observation_batch_tensor)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
    critic_loss.backward()
    critic_optimizer.step()

    # Train actor with PPO clipping
    with torch.no_grad():
        old_policy_action_probs = actor.forward(observation_batch_tensor)

    actor_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
        current_policy_action_probs = actor.forward(observation_batch_tensor)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        # Initialize policy
        self.policy = NNPolicy(self.actor)
        
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
            batch_actor_losses, batch_critic_losses = train_ppo(
                self.actor,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer,
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
        actor_lr=1e-4,
        critic_lr=1e-4,
        episodes_per_batch=32,
        train_epochs=1000,
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
