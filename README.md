# Multi-Environment Deep Reinforcement Learning Framework

A flexible Deep RL framework supporting multiple environment types with specialized DQN agents. Each environment has its own corresponding DQN agent optimized for that specific domain.

## 🧠 Architecture Overview

- **DQN-Only Framework**: All agents use Deep Q-Networks for scalability
- **Multi-Environment Support**: Extensible framework for various RL environments
- **Specialized Agents**: Each environment type has its own optimized DQN agent
- **Clean Factory Pattern**: Environment and agent creation through factory functions
- **Unified Interface**: Consistent API across all environment types

## 🌍 Supported Environments

### GridWorld Environment
- **Type**: `gridworld`
- **Description**: 2D grid navigation with rewards, enemies, and obstacles
- **Default Config**: 8x8 grid, 2 rewards, 4 enemies, 8 obstacles
- **State Space**: Agent position + grid encodings + distance vectors
- **Action Space**: 4 directions (UP, RIGHT, DOWN, LEFT)

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Train DQN Agent
```bash
python main.py train --env-type gridworld --episodes 1000
```

#### Test Trained Agent
```bash
python main.py test --env-type gridworld --model models/gridworld_dqnagent_final.pth --render
```

#### Human Play Mode
```bash
python main.py play --env-type gridworld
```

## 🏗️ Framework Architecture

### Class Hierarchy
```
BaseAgent (abstract)
└── DQNAgent (configurable for different environments)

Environment Factory
├── GridWorldEnv
└── [Future environments...]
```

### Agent Factory Pattern
```python
def create_agent(env_type: str, args) -> DQNAgent:
    """Factory function to create DQN agent based on environment type"""
    state_size = get_state_size(env_type, args)
    
    if env_type == "gridworld":
        return DQNAgent(state_size=state_size, ...)
    # Add more environment types here
```

## 🧮 DQN Agent Features

### Configurable Architecture
- **Hidden Layers**: Customizable network depth and width
- **State Processing**: Environment-specific input handling
- **Experience Replay**: Efficient memory buffer
- **Target Networks**: Stable learning with periodic updates

### Training Features
- **Epsilon-Greedy**: Configurable exploration strategy
- **Gradient Clipping**: Stable gradient updates
- **Periodic Saving**: Model checkpoints during training
- **Rich Metrics**: Loss tracking, epsilon decay, buffer monitoring

### Evaluation & Visualization
- **Performance Metrics**: Success rate, average rewards
- **Training Plots**: Rewards, episode lengths, loss curves
- **Model Persistence**: Complete state saving/loading

## 📊 GridWorld Environment Details

### Default Configuration (8x8)
- **Agent Start**: Top-left corner (0,0)
- **Rewards**: 2 rewards at strategic positions
- **Enemies**: 4 enemies to avoid
- **Obstacles**: 8 static obstacles blocking movement

### Reward System
- **Step Penalty**: -0.01 (efficiency incentive)
- **Reward Collection**: +10.0
- **Mission Complete**: +50.0 (all rewards collected)
- **Enemy Collision**: -20.0
- **Distance Reward**: Bonus for strategic positioning near enemies
- **Collision Penalty**: -0.05 (walls) / -0.1 (obstacles)

### State Representation
```
Observation Vector Components (Compact Version):
├── Agent Position (2D)
├── Distance Vectors to Rewards (num_rewards × 2) 
├── Distance Vectors to Enemies (num_enemies × 2)
└── Distance to Nearest Obstacles in 8 Directions (8D)

Total Size: 2 + num_rewards×2 + num_enemies×2 + 8
Default (2 rewards, 4 enemies): 2 + 4 + 8 + 8 = 22 dimensions
```

## 🛠️ Command Line Interface

### Global Options
- `--env-type {gridworld}` - Environment type selection
- `--eval-episodes N` - Episodes for evaluation

### Training Mode
```bash
python main.py train [options]
```

**Key Arguments:**
- `--episodes N` - Training episodes (default: 1000)
- `--learning-rate F` - Learning rate (default: 0.001)
- `--epsilon F` - Initial exploration rate (default: 1.0)
- `--buffer-size N` - Replay buffer size (default: 10000)
- `--batch-size N` - Training batch size (default: 64)
- `--plot` - Generate training plots
- `--evaluate` - Evaluate after training

### Testing Mode
```bash
python main.py test --model path/to/model.pth [options]
```

### Play Mode (Human Control)
```bash
python main.py play [options]
```
**Controls:** W/S/A/D or Arrow Keys, ESC to quit, R to reset

## 📁 Project Structure

```
RL/
├── agents/                    # DQN agent implementations
│   ├── base_agent.py         # Abstract base class
│   ├── dqn_agent.py          # Deep Q-Network agent
│   └── __init__.py
├── environment/               # Environment implementations
│   ├── grid_world.py         # GridWorld environment
│   └── __init__.py
├── main.py                   # CLI interface with factory patterns
├── models/                   # Saved models directory
│   └── [env_type]_dqnagent_final.pth
├── README.md                 # This file
└── requirements.txt          # Dependencies
```

## 🔧 Adding New Environments

### 1. Create Environment Class
```python
# environment/new_env.py
import gymnasium as gym

class NewEnv(gym.Env):
    def __init__(self, ...):
        # Environment-specific initialization
        pass
    
    def step(self, action):
        # Environment step logic
        pass
    
    def reset(self):
        # Reset environment
        pass
```

### 2. Update Factory Functions
```python
# main.py
def create_environment(args):
    if args.env_type == "newenv":
        return NewEnv(...)
    # ...

def get_state_size(env_type: str, args) -> int:
    if env_type == "newenv":
        return calculate_new_env_state_size(args)
    # ...

def create_agent(env_type: str, args) -> DQNAgent:
    if env_type == "newenv":
        return DQNAgent(
            state_size=get_state_size(env_type, args),
            hidden_sizes=[256, 128],  # Custom architecture
            ...
        )
    # ...
```

### 3. Add CLI Arguments
```python
# main.py - Add environment-specific arguments
parser.add_argument('--newenv-param', ...)
```

## 📈 Performance Optimization

### Network Architecture
- **Adaptive Hidden Sizes**: Configure based on state complexity
- **GPU Support**: Automatic CUDA detection and usage
- **Gradient Clipping**: Stable training with large networks

### Training Efficiency
- **Experience Replay**: Sample efficiency through memory replay
- **Target Networks**: Stable Q-learning with periodic updates
- **Batch Processing**: Efficient GPU utilization

### Compact State Representation
- **Distance Vectors**: Use relative positions instead of full grids
- **Directional Obstacles**: 8-direction obstacle sensing vs full grid
- **Dramatic Size Reduction**: From 202 to 22 dimensions (90% reduction)
- **Faster Training**: Smaller networks, faster forward/backward passes

## 🎯 Future Roadmap

- [ ] **Classic Control**: CartPole, MountainCar, Pendulum
- [ ] **Atari Games**: Visual RL with CNN-based DQNs
- [ ] **Continuous Control**: DDPG/TD3 for continuous action spaces
- [ ] **Multi-Agent**: Cooperative and competitive scenarios
- [ ] **Custom Domains**: Domain-specific environments

## 🔬 Research Extensions

- **Double DQN**: Reduce overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions
- **Rainbow DQN**: Combine multiple DQN improvements

## 💡 Design Philosophy

1. **Modularity**: Clean separation between environments and agents
2. **Extensibility**: Easy addition of new environments and agent types
3. **Consistency**: Unified interface across all components
4. **Scalability**: Deep learning approach suitable for complex domains
5. **Reproducibility**: Fixed seeds and deterministic training