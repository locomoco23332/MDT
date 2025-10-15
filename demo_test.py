"""
Demo script showing how to test a trained PPO model

This script demonstrates different ways to test and evaluate a trained PPO model.
"""

import os
import sys
from test_ppo_model import PPOModelTester


def demo_basic_testing():
    """Demonstrate basic model testing"""
    print("=== Basic Model Testing Demo ===")
    
    model_path = "ppo_metadrive_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using ppo_metadrive.py")
        return
    
    try:
        # Create tester
        print("📦 Loading model...")
        tester = PPOModelTester(model_path)
        
        # Test single episode
        print("\n🎮 Testing single episode...")
        episode_stats = tester.test_single_episode(render=True)
        
        print(f"Episode completed with reward: {episode_stats['total_reward']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_multiple_episodes():
    """Demonstrate testing multiple episodes"""
    print("\n=== Multiple Episodes Testing Demo ===")
    
    model_path = "ppo_metadrive_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Create tester
        tester = PPOModelTester(model_path)
        
        # Test multiple episodes
        print("🎮 Testing multiple episodes...")
        episode_stats = tester.test_multiple_episodes(num_episodes=3, render=True)
        
        # Plot performance
        print("📊 Generating performance plots...")
        tester.plot_performance(episode_stats)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_video_recording():
    """Demonstrate video recording"""
    print("\n=== Video Recording Demo ===")
    
    model_path = "ppo_metadrive_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Create tester
        tester = PPOModelTester(model_path)
        
        # Test with video recording
        print("🎥 Recording video...")
        episode_stats = tester.test_single_episode(render=True, save_video=True)
        
        if episode_stats['frames']:
            print("💾 Saving video...")
            tester.save_video(episode_stats['frames'], "demo_ppo_test.mp4")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_interactive_mode():
    """Demonstrate interactive testing mode"""
    print("\n=== Interactive Testing Demo ===")
    
    model_path = "ppo_metadrive_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Create tester
        tester = PPOModelTester(model_path)
        
        print("🎮 Starting interactive mode...")
        print("You can now test the model interactively!")
        tester.interactive_test()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print("🚗 PPO MetaDrive Model Testing Demo")
    print("=" * 50)
    
    # Check if model exists
    model_path = "ppo_metadrive_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\nTo create a model, run:")
        print("  python3 ppo_metadrive.py")
        print("\nOr use the example usage script:")
        print("  python3 example_usage.py")
        return
    
    print("Available demos:")
    print("1. Basic testing (single episode)")
    print("2. Multiple episodes testing")
    print("3. Video recording")
    print("4. Interactive mode")
    print("5. Run all demos")
    
    try:
        choice = input("\nSelect demo (1-5): ").strip()
        
        if choice == "1":
            demo_basic_testing()
        elif choice == "2":
            demo_multiple_episodes()
        elif choice == "3":
            demo_video_recording()
        elif choice == "4":
            demo_interactive_mode()
        elif choice == "5":
            demo_basic_testing()
            demo_multiple_episodes()
            demo_video_recording()
        else:
            print("Invalid choice. Running basic demo...")
            demo_basic_testing()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()

