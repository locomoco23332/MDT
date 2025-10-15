"""
Test script to verify model saving functionality

This script creates a minimal training session to test that models are saved correctly.
"""

import os
import torch
from ppo_metadrive import PPOTrainer, PPOConfig


def test_model_saving():
    """Test that models are saved correctly during training"""
    
    print("🧪 Testing PPO Model Saving Functionality")
    print("=" * 50)
    
    # Create a minimal configuration for quick testing
    config = PPOConfig(
        train_epochs=5,           # Very short training
        episodes_per_batch=2,     # Small batch size
        save_frequency=2,         # Save every 2 steps
        save_final_model=True,    # Save final model
        horizon=100,              # Short episodes
        num_scenarios=10          # Few scenarios
    )
    
    print(f"Configuration: {config}")
    
    # Create trainer
    trainer = PPOTrainer(config)
    
    print("\n🚀 Starting minimal training to test saving...")
    
    try:
        # Train (this should save models automatically)
        trainer.train()
        
        print("\n✅ Training completed!")
        
        # Check if models were saved
        expected_files = [
            "ppo_metadrive_model.pth",           # Final model
            "ppo_metadrive_best_model.pth",      # Best model
        ]
        
        # Check for periodic saves (step 2 and 4)
        for step in [2, 4]:
            expected_files.append(f"ppo_metadrive_model_step_{step}.pth")
        
        print("\n📁 Checking saved model files:")
        
        all_saved = True
        for filename in expected_files:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"  ✅ {filename} ({file_size:,} bytes)")
            else:
                print(f"  ❌ {filename} - NOT FOUND")
                all_saved = False
        
        if all_saved:
            print("\n🎉 All model files saved successfully!")
            print("\n📋 Summary of saved models:")
            print("  - ppo_metadrive_model.pth: Final trained model")
            print("  - ppo_metadrive_best_model.pth: Best performing model")
            print("  - ppo_metadrive_model_step_X.pth: Periodic saves")
            
            return True
        else:
            print("\n❌ Some model files are missing!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test that saved models can be loaded correctly"""
    
    print("\n🔍 Testing Model Loading...")
    
    model_files = [
        "ppo_metadrive_model.pth",
        "ppo_metadrive_best_model.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                # Try to load the model
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Check if required keys exist
                required_keys = ['actor_state_dict', 'critic_state_dict', 'config']
                missing_keys = [key for key in required_keys if key not in checkpoint]
                
                if missing_keys:
                    print(f"  ❌ {model_file}: Missing keys {missing_keys}")
                else:
                    print(f"  ✅ {model_file}: Loaded successfully")
                    
            except Exception as e:
                print(f"  ❌ {model_file}: Error loading - {e}")
        else:
            print(f"  ⚠️  {model_file}: File not found")


def cleanup_test_files():
    """Clean up test files"""
    
    print("\n🧹 Cleaning up test files...")
    
    test_files = [
        "ppo_metadrive_model.pth",
        "ppo_metadrive_best_model.pth",
        "ppo_metadrive_model_step_2.pth",
        "ppo_metadrive_model_step_4.pth"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"  🗑️  Removed {filename}")
            except Exception as e:
                print(f"  ❌ Error removing {filename}: {e}")


def main():
    """Main test function"""
    
    print("This script will test the model saving functionality with a short training session.")
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Test cancelled.")
        return
    
    # Test model saving
    success = test_model_saving()
    
    if success:
        # Test model loading
        test_model_loading()
        
        # Ask if user wants to keep the models
        keep_models = input("\nKeep the saved models? (y/n): ").strip().lower()
        if keep_models != 'y':
            cleanup_test_files()
        else:
            print("Models kept for further testing.")
    else:
        print("Model saving test failed.")


if __name__ == "__main__":
    main()

