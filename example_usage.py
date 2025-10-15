"""
Example usage of PPO algorithm for MetaDrive environment

This script demonstrates how to use the PPO implementation
to train an autonomous driving agent in MetaDrive.
"""

import torch
import numpy as np
from ppo_metadrive import PPOTrainer, PPOConfig


def quick_training_example():
    """Quick training example with default parameters"""
    print("=== Quick PPO Training Example ===")
    
    # Use default configuration
    trainer = PPOTrainer()
    
    # Train for a few epochs (reduced for demo)
    trainer.config.train_epochs = 10
    trainer.config.episodes_per_batch = 8
    
    print("Starting training...")
    trainer.train()
    
    # Evaluate the trained policy
    print("\nEvaluating trained policy...")
    trainer.evaluate(num_episodes=3, render=True)
    
    return trainer


def custom_config_example():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = PPOConfig(
        ppo_eps=0.1,                    # Smaller clipping parameter
        ppo_grad_descent_steps=5,       # Fewer gradient steps
        gamma=0.95,                     # Discount factor
        actor_lr=1e-4,                  # Actor learning rate
        critic_lr=5e-4,                 # Critic learning rate
        episodes_per_batch=16,          # Batch size
        train_epochs=20,                # Training epochs
        horizon=200,                    # Episode length
        num_scenarios=50                # Number of scenarios
    )
    
    trainer = PPOTrainer(custom_config)
    
    print(f"Using custom config: {custom_config}")
    trainer.train()
    
    return trainer


def save_and_load_example():
    """Example of saving and loading models"""
    print("\n=== Save and Load Example ===")
    
    # Train a model
    trainer = PPOTrainer()
    trainer.config.train_epochs = 5
    trainer.config.episodes_per_batch = 4
    trainer.train()
    
    # Save the model
    model_path = "example_ppo_model.pth"
    trainer.save_model(model_path)
    
    # Create a new trainer and load the model
    new_trainer = PPOTrainer()
    new_trainer.load_model(model_path)
    
    # Evaluate the loaded model
    print("Evaluating loaded model...")
    new_trainer.evaluate(num_episodes=2, render=True)
    
    return new_trainer


def hyperparameter_comparison():
    """Compare different hyperparameter settings"""
    print("\n=== Hyperparameter Comparison ===")
    
    configs = [
        ("Conservative", PPOConfig(
            ppo_eps=0.1,
            actor_lr=1e-4,
            critic_lr=1e-4,
            train_epochs=5,
            episodes_per_batch=4
        )),
        ("Aggressive", PPOConfig(
            ppo_eps=0.3,
            actor_lr=1e-3,
            critic_lr=1e-3,
            train_epochs=5,
            episodes_per_batch=4
        ))
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTraining with {name} configuration...")
        trainer = PPOTrainer(config)
        trainer.train()
        
        # Evaluate
        avg_reward = trainer.evaluate(num_episodes=3, render=False)
        results[name] = avg_reward
        
        print(f"{name} configuration achieved average reward: {avg_reward:.3f}")
    
    # Compare results
    print("\n=== Comparison Results ===")
    for name, reward in results.items():
        print(f"{name}: {reward:.3f}")
    
    best_config = max(results.items(), key=lambda x: x[1])
    print(f"Best configuration: {best_config[0]} with reward {best_config[1]:.3f}")


def monitoring_example():
    """Example with training monitoring and visualization"""
    print("\n=== Training Monitoring Example ===")
    
    trainer = PPOTrainer()
    trainer.config.train_epochs = 15
    trainer.config.episodes_per_batch = 8
    
    # Train with monitoring
    trainer.train()
    
    # Plot training progress
    print("Generating training plots...")
    trainer.plot_training_progress()
    
    return trainer


def main():
    """Main function to run all examples"""
    print("PPO MetaDrive Examples")
    print("=" * 50)
    
    try:
        # Run examples
        quick_training_example()
        custom_config_example()
        save_and_load_example()
        hyperparameter_comparison()
        monitoring_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install torch gymnasium metadrive matplotlib numpy")


if __name__ == "__main__":
    main()



