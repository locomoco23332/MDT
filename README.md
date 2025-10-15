# PPO Algorithm for MetaDrive Environment

This implementation provides a complete Proximal Policy Optimization (PPO) algorithm for training autonomous driving agents in the MetaDrive environment.

## Features

- **Complete PPO Implementation**: Actor-Critic architecture with PPO clipping
- **MetaDrive Integration**: Ready-to-use with MetaDrive autonomous driving environment
- **Flexible Configuration**: Customizable hyperparameters and training settings
- **Training Monitoring**: Built-in visualization and progress tracking
- **Model Persistence**: Save and load trained models
- **GPU Support**: Automatic GPU detection and usage

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch gymnasium metadrive matplotlib numpy
```

### Basic Usage

```python
from ppo_metadrive import PPOTrainer, PPOConfig

# Create trainer with default configuration
trainer = PPOTrainer()

# Train the agent
trainer.train()

# Evaluate the trained policy
trainer.evaluate(num_episodes=5, render=True)

# Plot training progress
trainer.plot_training_progress()
```

### Custom Configuration

```python
# Create custom configuration
config = PPOConfig(
    ppo_eps=0.2,                    # PPO clipping parameter
    gamma=0.99,                     # Discount factor
    actor_lr=3e-4,                  # Actor learning rate
    critic_lr=1e-3,                 # Critic learning rate
    episodes_per_batch=32,          # Batch size
    train_epochs=100,               # Training epochs
    horizon=300,                    # Episode length
    num_scenarios=100               # Number of scenarios
)

trainer = PPOTrainer(config)
trainer.train()
```

## Architecture

### Networks

- **Actor Network**: Policy network that outputs continuous actions (throttle, steering)
- **Critic Network**: Value network that estimates state values
- **Architecture**: CNN-based with 2 convolutional layers + 2 fully connected layers

### PPO Algorithm

- **Clipped Surrogate Loss**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Computes advantages
- **Dual Optimizers**: Separate optimizers for actor and critic
- **Batch Training**: Collects multiple episodes per training batch

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ppo_eps` | 0.2 | PPO clipping parameter |
| `ppo_grad_descent_steps` | 10 | Number of gradient descent steps per batch |
| `gamma` | 0.99 | Discount factor for rewards |
| `lambda_gae` | 0.95 | GAE parameter |
| `actor_lr` | 3e-4 | Actor learning rate |
| `critic_lr` | 1e-3 | Critic learning rate |
| `episodes_per_batch` | 32 | Number of episodes per training batch |
| `train_epochs` | 100 | Total training epochs |
| `horizon` | 300 | Maximum episode length |
| `num_scenarios` | 100 | Number of scenarios in environment |

## Examples

### Basic Training

```python
from ppo_metadrive import PPOTrainer

trainer = PPOTrainer()
trainer.train()
```

### Custom Configuration

```python
from ppo_metadrive import PPOTrainer, PPOConfig

config = PPOConfig(
    train_epochs=50,
    episodes_per_batch=16,
    actor_lr=1e-4
)

trainer = PPOTrainer(config)
trainer.train()
```

### Model Persistence

```python
# Save model
trainer.save_model("my_ppo_model.pth")

# Load model
new_trainer = PPOTrainer()
new_trainer.load_model("my_ppo_model.pth")
```

### Evaluation

```python
# Evaluate with rendering
trainer.evaluate(num_episodes=5, render=True)

# Evaluate without rendering (faster)
avg_reward = trainer.evaluate(num_episodes=10, render=False)
```

## Training Output

The training process provides detailed progress information:

```
Step 0, Avg. Returns: 1.551 +/- 0.331, Median: 1.569, Actor Loss: -0.055, Critic Loss: 0.004
Step 1, Avg. Returns: 1.771 +/- 0.438, Median: 1.714, Actor Loss: 0.494, Critic Loss: 0.246
...
```

## Visualization

The implementation includes comprehensive visualization tools:

- **Training Returns**: Mean, median, and standard deviation of returns
- **Loss Curves**: Actor and critic loss over training
- **Return Distribution**: Scatter plot of individual episode returns

## File Structure

```
PPO/
├── ppo_metadrive.py      # Main PPO implementation
├── example_usage.py      # Usage examples
├── requirements.txt      # Dependencies
├── README.md            # This file
└── ppo_solution.ipynb   # Original solution notebook
```

## Dependencies

- **PyTorch**: Deep learning framework
- **Gymnasium**: RL environment interface
- **MetaDrive**: Autonomous driving environment
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

## GPU Support

The implementation automatically detects and uses GPU if available:

```python
# Check device
print(f"Training on: {trainer.device}")

# Force CPU usage
trainer.device = torch.device("cpu")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `episodes_per_batch` or `horizon`
2. **Slow Training**: Ensure GPU is being used, reduce batch size
3. **Poor Performance**: Adjust learning rates or PPO parameters

### Performance Tips

- Use GPU for faster training
- Adjust batch size based on available memory
- Monitor loss curves for training stability
- Experiment with different learning rates

## License

This implementation is based on the original PPO solution and follows the same licensing terms as the parent project.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.



