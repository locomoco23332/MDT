# PPO MetaDrive Model Testing

This directory contains comprehensive testing scripts for trained PPO models in the MetaDrive environment.

## 🚀 Quick Start

### Test if you have a model:
```bash
python3 test_with_model.py
```

### Quick test (3 episodes):
```bash
python3 quick_test.py
```

### Comprehensive testing:
```bash
python3 test_ppo_model.py --episodes 5
```

## 📁 Testing Scripts

### 1. `test_with_model.py` - Simple Model Checker
- **Purpose**: Check if a model exists and test it with one episode
- **Usage**: `python3 test_with_model.py`
- **Features**: 
  - Checks for model file existence
  - Provides instructions if no model found
  - Runs single episode test with rendering

### 2. `quick_test.py` - Quick Testing
- **Purpose**: Fast testing of trained models
- **Usage**: `python3 quick_test.py --episodes 3`
- **Features**:
  - Command-line arguments
  - Multiple episode testing
  - Performance summary
  - Simple and fast

### 3. `test_ppo_model.py` - Comprehensive Testing
- **Purpose**: Full-featured testing with advanced options
- **Usage**: `python3 test_ppo_model.py --episodes 5 --video`
- **Features**:
  - Model loading and validation
  - Multiple testing modes
  - Video recording
  - Performance plotting
  - Interactive mode
  - Detailed statistics

### 4. `demo_test.py` - Interactive Demo
- **Purpose**: Demonstrate different testing capabilities
- **Usage**: `python3 demo_test.py`
- **Features**:
  - Menu-driven interface
  - Multiple demo modes
  - Educational examples

## 🎮 Testing Modes

### Basic Testing
```bash
# Test single episode
python3 test_ppo_model.py --episodes 1

# Test multiple episodes
python3 test_ppo_model.py --episodes 5
```

### Video Recording
```bash
# Record video of first episode
python3 test_ppo_model.py --video

# Test with video recording
python3 test_ppo_model.py --episodes 3 --video
```

### No Rendering (Faster)
```bash
# Test without rendering (faster)
python3 test_ppo_model.py --episodes 10 --no-render
```

### Interactive Mode
```bash
# Interactive testing
python3 test_ppo_model.py --interactive
```

## 📊 Output and Results

### Console Output
- Real-time episode progress
- Step-by-step rewards
- Episode completion status
- Summary statistics

### Performance Metrics
- Total reward per episode
- Episode length (steps)
- Success rate
- Average performance
- Best/worst episodes

### Visualizations
- Episode reward plots
- Episode length plots
- Success rate trends
- Reward distribution histograms

### Video Output
- MP4 video files of episodes
- Individual frame images
- Configurable frame rate

## 🔧 Command Line Options

### `test_ppo_model.py` Options:
- `--model PATH`: Path to model file (default: ppo_metadrive_model.pth)
- `--episodes N`: Number of episodes to test (default: 5)
- `--no-render`: Disable rendering for faster testing
- `--video`: Record video of first episode
- `--interactive`: Run in interactive mode

### `quick_test.py` Options:
- `--model PATH`: Path to model file (default: ppo_metadrive_model.pth)
- `--episodes N`: Number of episodes to test (default: 3)

## 📈 Performance Evaluation

### Success Criteria:
- **Excellent**: Average reward > 50
- **Good**: Average reward > 20
- **Needs Improvement**: Average reward < 20

### Metrics Tracked:
- Total episode reward
- Episode length (steps)
- Success rate (episodes completed without timeout)
- Reward consistency (standard deviation)

## 🎥 Video Recording

### Requirements:
- OpenCV (`pip install opencv-python`) for MP4 videos
- Matplotlib for image sequences

### Output Formats:
- MP4 video files
- Individual PNG images
- Configurable resolution and frame rate

## 🐛 Troubleshooting

### Common Issues:

1. **Model not found**:
   ```
   ❌ Model file not found: ppo_metadrive_model.pth
   ```
   **Solution**: Train a model first using `ppo_metadrive.py`

2. **Import errors**:
   ```
   ModuleNotFoundError: No module named 'metadrive'
   ```
   **Solution**: Install MetaDrive properly (see main README)

3. **Rendering issues**:
   ```
   Display not available
   ```
   **Solution**: Use `--no-render` flag for headless testing

4. **Memory issues**:
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce episode length or use CPU

### Performance Tips:
- Use `--no-render` for faster testing
- Reduce episode length for quick tests
- Use fewer episodes for initial testing
- Monitor GPU memory usage

## 📝 Example Usage

### Basic Testing Workflow:
```bash
# 1. Check if model exists
python3 test_with_model.py

# 2. Quick test
python3 quick_test.py --episodes 3

# 3. Comprehensive test
python3 test_ppo_model.py --episodes 5

# 4. Record video
python3 test_ppo_model.py --video
```

### Interactive Testing:
```bash
# Start interactive mode
python3 test_ppo_model.py --interactive

# Available commands:
# - test: Test single episode
# - multi: Test multiple episodes
# - video: Test with video recording
# - quit: Exit
```

## 🔗 Integration

These testing scripts integrate with:
- `ppo_metadrive.py`: Main training script
- `example_usage.py`: Training examples
- MetaDrive environment: Autonomous driving simulation
- PyTorch models: Actor-Critic networks

## 📚 Additional Resources

- Main PPO implementation: `ppo_metadrive.py`
- Training examples: `example_usage.py`
- Requirements: `requirements.txt`
- Main documentation: `README.md`

