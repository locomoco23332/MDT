"""
Simple test script that works with any trained PPO model

This script will test a PPO model if it exists, or provide instructions
on how to create one.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from ppo_metadrive import Actor, Critic, NNPolicy


def test_model_if_exists():
    """Test model if it exists, otherwise provide instructions"""
    
    model_path = "ppo_metadrive_model.pth"
    
    print("🚗 PPO MetaDrive Model Tester")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("\n📝 To create a model, you have several options:")
        print("\n1. Train a new model:")
        print("   python3 ppo_metadrive.py")
        print("\n2. Use the example script:")
        print("   python3 example_usage.py")
        print("\n3. Quick training (fewer epochs):")
        print("   python3 -c \"from ppo_metadrive import PPOTrainer; trainer = PPOTrainer(); trainer.config.train_epochs = 10; trainer.train()\"")
        
        return False
    
    print(f"✅ Found model: {model_path}")
    
    try:
        # Load model
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
        
        # Test the model
        print("\n🎮 Testing model in MetaDrive environment...")
        
        # Create environment
        env = gym.make(
            "MetaDrive-topdown",
            config={
                "use_render": True,
                "horizon": 300,
                "num_scenarios": 100
            }
        )
        
        # Run one episode
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        print("🚀 Starting episode...")
        
        while not done:
            # Get action from policy
            action = policy(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 50 == 0:
                print(f"  Step {step_count}, Reward: {total_reward:.2f}")
        
        env.close()
        
        # Results
        print(f"\n🎯 Episode Results:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps: {step_count}")
        print(f"   Success: {not truncated}")
        
        if total_reward > 50:
            print("🎉 Excellent performance!")
        elif total_reward > 20:
            print("👍 Good performance!")
        else:
            print("📈 Model needs more training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    success = test_model_if_exists()
    
    if success:
        print("\n🎮 To run more comprehensive tests:")
        print("   python3 test_ppo_model.py --episodes 5")
        print("   python3 quick_test.py --episodes 3")
        print("   python3 demo_test.py")
    else:
        print("\n💡 Once you have a model, you can test it with:")
        print("   python3 test_with_model.py")


if __name__ == "__main__":
    main()

