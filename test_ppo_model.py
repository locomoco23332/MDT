"""
Test script for trained PPO MetaDrive model

This script loads a trained PPO model and renders it in the MetaDrive environment
for visualization and evaluation purposes.
"""

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import os
from typing import List, Tuple, Optional
from ppo_metadrive import PPOTrainer, PPOConfig, Actor, Critic, NNPolicy, obs_batch_to_tensor, deviceof


class PPOModelTester:
    """Test class for trained PPO models"""
    
    def __init__(self, model_path: str, config: Optional[PPOConfig] = None):
        """
        Initialize the tester with a trained model
        
        Args:
            model_path: Path to the saved model (.pth file)
            config: PPO configuration (optional, will be loaded from model if not provided)
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.load_model(model_path, config)
        
        # Initialize policy
        self.policy = NNPolicy(self.actor)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {self.device}")

    def load_model(self, model_path: str, config: Optional[PPOConfig] = None):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        if config is None:
            self.config = checkpoint.get('config', PPOConfig())
        else:
            self.config = config
        
        # Initialize networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        
        # Load state dicts
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        print("Model loaded successfully!")

    def create_env(self, render: bool = True, horizon: int = 500, num_scenarios: int = 100) -> gym.Env:
        """Create MetaDrive environment for testing"""
        return gym.make(
            "MetaDrive-topdown",
            config={
                "use_render": render,
                "horizon": horizon,
                "num_scenarios": num_scenarios
            }
        )

    def test_single_episode(self, render: bool = True, save_video: bool = False, 
                          episode_id: int = 0) -> dict:
        """
        Test the model on a single episode
        
        Args:
            render: Whether to render the environment
            save_video: Whether to save video frames
            episode_id: Episode identifier for saving
            
        Returns:
            Dictionary with episode statistics
        """
        env = self.create_env(render=render, horizon=500)
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        frames = [] if save_video else None
        
        print(f"Starting episode {episode_id}...")
        
        while not done:
            # Get action from policy
            action = self.policy(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Save frame if requested
            if save_video and render:
                frames.append(env.render())
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}, Reward: {total_reward:.2f}")
        
        env.close()
        
        episode_stats = {
            'total_reward': total_reward,
            'step_count': step_count,
            'success': not truncated,  # Episode completed without timeout
            'frames': frames
        }
        
        print(f"Episode {episode_id} completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Steps: {step_count}")
        print(f"Success: {episode_stats['success']}")
        
        return episode_stats

    def test_multiple_episodes(self, num_episodes: int = 5, render: bool = True) -> List[dict]:
        """
        Test the model on multiple episodes
        
        Args:
            num_episodes: Number of episodes to test
            render: Whether to render the environment
            
        Returns:
            List of episode statistics
        """
        print(f"Testing model on {num_episodes} episodes...")
        
        all_episodes = []
        total_rewards = []
        
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            episode_stats = self.test_single_episode(render=render, episode_id=i+1)
            all_episodes.append(episode_stats)
            total_rewards.append(episode_stats['total_reward'])
            
            # Small delay between episodes
            time.sleep(1)
        
        # Print summary statistics
        print(f"\n=== Test Results Summary ===")
        print(f"Number of episodes: {num_episodes}")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Best reward: {np.max(total_rewards):.2f}")
        print(f"Worst reward: {np.min(total_rewards):.2f}")
        print(f"Success rate: {np.mean([ep['success'] for ep in all_episodes]):.2%}")
        
        return all_episodes

    def save_video(self, frames: List, filename: str = "ppo_test_video.mp4"):
        """Save video frames to file"""
        if not frames:
            print("No frames to save")
            return
        
        try:
            import cv2
            
            # Get frame dimensions
            height, width, channels = frames[0].shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved as {filename}")
            
        except ImportError:
            print("OpenCV not available. Saving frames as images instead...")
            self.save_frames_as_images(frames, filename.replace('.mp4', ''))

    def save_frames_as_images(self, frames: List, prefix: str = "frame"):
        """Save frames as individual images"""
        for i, frame in enumerate(frames):
            plt.figure(figsize=(10, 10))
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"Frame {i}")
            plt.savefig(f"{prefix}_{i:04d}.png", bbox_inches='tight', dpi=100)
            plt.close()
        
        print(f"Saved {len(frames)} frames as images with prefix '{prefix}'")

    def plot_performance(self, episode_stats: List[dict], save_plot: bool = True):
        """Plot performance metrics"""
        rewards = [ep['total_reward'] for ep in episode_stats]
        steps = [ep['step_count'] for ep in episode_stats]
        successes = [ep['success'] for ep in episode_stats]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward plot
        axes[0, 0].plot(rewards, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Steps plot
        axes[0, 1].plot(steps, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        success_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        axes[1, 0].plot(success_rate, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Cumulative Success Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 1].hist(rewards, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('ppo_test_performance.png', dpi=300, bbox_inches='tight')
            print("Performance plot saved as 'ppo_test_performance.png'")
        
        plt.show()

    def interactive_test(self):
        """Interactive testing mode"""
        print("=== Interactive PPO Model Testing ===")
        print("Commands:")
        print("  'test' - Test single episode")
        print("  'multi' - Test multiple episodes")
        print("  'video' - Test with video recording")
        print("  'quit' - Exit")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'test':
                self.test_single_episode(render=True)
            elif command == 'multi':
                try:
                    num_episodes = int(input("Number of episodes (default 5): ") or "5")
                    self.test_multiple_episodes(num_episodes, render=True)
                except ValueError:
                    print("Invalid number, using default 5 episodes")
                    self.test_multiple_episodes(5, render=True)
            elif command == 'video':
                print("Recording video...")
                episode_stats = self.test_single_episode(render=True, save_video=True)
                if episode_stats['frames']:
                    self.save_video(episode_stats['frames'])
            else:
                print("Unknown command. Available: test, multi, video, quit")


def main():
    """Main function for testing PPO model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained PPO MetaDrive model')
    parser.add_argument('--model', type=str, default='ppo_metadrive_best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to test')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (faster testing)')
    parser.add_argument('--video', action='store_true',
                       help='Record video of first episode')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train a model first using ppo_metadrive.py")
        return
    
    try:
        # Create tester
        tester = PPOModelTester(args.model)
        
        if args.interactive:
            # Interactive mode
            tester.interactive_test()
        else:
            # Standard testing
            if args.video:
                # Test with video recording
                print("Testing with video recording...")
                episode_stats = tester.test_single_episode(
                    render=not args.no_render, 
                    save_video=True
                )
                if episode_stats['frames']:
                    tester.save_video(episode_stats['frames'])
            else:
                # Test multiple episodes
                episode_stats = tester.test_multiple_episodes(
                    num_episodes=args.episodes,
                    render=not args.no_render
                )
                
                # Plot performance
                tester.plot_performance(episode_stats)
    
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

