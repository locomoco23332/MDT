"""
Quick test script for PPO MetaDrive model

Simple script to quickly test a trained PPO model with rendering.
"""

import torch
import numpy as np
import gymnasium as gym
import os
from ppo_metadrive import PPOTrainer, PPOConfig, Actor, Critic, NNPolicy


def quick_test(model_path: str = "ppo_metadrive_model.pth", num_episodes: int = 3):
    """
    Quick test of a trained PPO model
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
    """
    print(f"Testing PPO model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using ppo_metadrive.py")
        return
    
    try:
        # Load the model
        print("📦 Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create networks
        actor = Actor()
        critic = Critic()
        
        # Load weights
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Set to evaluation mode
        actor.eval()
        critic.eval()
        
        # Create policy
        policy = NNPolicy(actor)
        
        print("✅ Model loaded successfully!")
        
        # Create environment
        print("🚗 Creating MetaDrive environment...")
        env = gym.make(
            "MetaDrive-topdown",
            config={
                "use_render": True,
                "horizon": 500,
                "num_scenarios": 100
            }
        )
        
        print(f"🎮 Testing {num_episodes} episodes...")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            while not done:
                # Get action from policy
                action = policy(obs)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"  Step {step_count}, Reward: {total_reward:.2f}")
            
            total_rewards.append(total_reward)
            print(f"✅ Episode {episode + 1} completed!")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Steps: {step_count}")
            print(f"   Success: {not truncated}")
        
        env.close()
        
        # Print summary
        print(f"\n🎯 Test Results Summary:")
        print(f"   Episodes tested: {num_episodes}")
        print(f"   Average reward: {np.mean(total_rewards):.2f}")
        print(f"   Best reward: {np.max(total_rewards):.2f}")
        print(f"   Worst reward: {np.min(total_rewards):.2f}")
        print(f"   Reward std: {np.std(total_rewards):.2f}")
        
        if np.mean(total_rewards) > 50:
            print("🎉 Great performance!")
        elif np.mean(total_rewards) > 20:
            print("👍 Good performance!")
        else:
            print("📈 Room for improvement!")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test of PPO MetaDrive model')
    parser.add_argument('--model', type=str, default='ppo_metadrive_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to test')
    
    args = parser.parse_args()
    
    quick_test(args.model, args.episodes)


if __name__ == "__main__":
    main()

