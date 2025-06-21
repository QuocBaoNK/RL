# Deep Q-Network (DQN) for GridWorld Navigation

A reinforcement learning implementation using Deep Q-Network (DQN) to train agents for navigation in GridWorld environments.

## Overview

This project implements an intelligent agent that learns to navigate a 2D grid to collect rewards while avoiding enemies and obstacles. The agent uses the DQN algorithm to learn from experience and improve its strategy over time.

## Features

- **DQN Agent**: Deep Q-Network with experience replay and target networks
- **Multiple Environments**: Static and dynamic GridWorld variants
- **Reward Shaping**: Potential-based reward shaping for faster convergence
- **Training Pipeline**: Complete training and evaluation framework
- **Interactive Play**: Human control mode for testing

## Environments

### GridWorld
- 8x8 grid with agent, rewards, enemies, and obstacles
- Static enemies at fixed positions
- Agent must collect all rewards to complete the mission

### Moving Enemy GridWorld  
- Similar to basic GridWorld but with moving enemies
- More challenging due to dynamic environment
- Enemies have 80% probability to move each step

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
# Train on standard GridWorld
python main.py train --episodes 1000

# Train on GridWorld with moving enemies
python main.py train --env-type movingenemyworld --episodes 1000
```

### Testing
```bash
# Test with visualization
python main.py test --render

# Test on moving enemy environment
python main.py test --env-type movingenemyworld --render
```

### Human Play
```bash
# Control agent with W/A/S/D keys
python main.py play
```

## How It Works

### DQN Algorithm
- Uses neural networks to estimate Q-values
- Experience replay for learning from past experiences
- Target network for stable training
- Epsilon-greedy strategy for exploration/exploitation balance

### Reward System
- Reward collection: +10 points
- Mission completion: +50 points
- Enemy collision: -20 points (episode ends)
- Step penalty: -0.01 points (encourages efficiency)
- Wall/obstacle collision: -0.05/-0.1 points

### Reward Shaping
Implemented potential-based reward shaping for faster learning:
- Calculates distance to nearest reward
- Provides bonus when agent moves closer to rewards
- Uses formula: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
- Results in 2-4x faster convergence compared to standard rewards

## Performance Results

### Reward Shaping Comparison (100 episodes, 5x5 grid):
| Metric | With Shaping | Without Shaping | Improvement |
|--------|--------------|-----------------|-------------|
| Average Reward | 36.90 | 10.63 | **+247%** |
| Episode Length | 8.3 steps | 74.8 steps | **-88%** |
| Training Time | 5.76s | 30.78s | **-81%** |

### Expected Performance
- **GridWorld**: 80-95% success rate after 1000 episodes
- **Moving Enemy**: 70-85% success rate (more challenging)
- **Training Time**: 10-20 minutes depending on environment

## Project Structure

```
RL/
├── agents/
│   ├── base_agent.py         # Base agent class
│   └── dqn_agent.py          # DQN agent implementation
├── environment/
│   ├── grid_world.py         # Basic GridWorld environment
│   └── moving_enemy_grid_world.py  # GridWorld with moving enemies
├── main.py                   # Main entry point
├── models/                   # Saved model directory
└── requirements.txt          # Dependencies
```

## Configuration Options

### Training Parameters
- `--episodes N`: Number of training episodes (default: 1000)
- `--learning-rate F`: Learning rate (default: 0.001)
- `--epsilon F`: Initial exploration rate (default: 1.0)
- `--batch-size N`: Batch size (default: 64)

### Environment Parameters
- `--env-type TYPE`: Environment type (gridworld/movingenemyworld)
- `--grid-size N`: Grid size NxN (default: 8)
- `--num-rewards N`: Number of rewards (default: 2)
- `--num-enemies N`: Number of enemies (default: 4)
- `--use-reward-shaping`: Enable reward shaping (default)
- `--no-reward-shaping`: Disable reward shaping for comparison

## Key Implementation Details

### Neural Network Architecture
- Fully connected layers with ReLU activation
- Default hidden layers: [128, 64]
- Adaptive to different state space sizes

### Training Features
- Experience replay buffer (default: 10,000)
- Target network updates every 100 steps
- Epsilon decay from 1.0 to 0.01
- Gradient clipping for stability

### State Representation
The agent observes a 22-dimensional state vector containing:
- Agent position (2D)
- Distance vectors to rewards
- Distance vectors to enemies
- Obstacle distance measurements

## Advanced Features

### Reward Shaping Theory
Implements potential-based reward shaping which:
- Maintains optimal policy guarantees
- Accelerates learning without changing optimal behavior
- Uses Manhattan distance as potential function
- Provides dense reward signal for sparse reward environments

### Multi-Environment Support
- Modular environment design
- Easy to extend with new environment types
- Consistent interface across environments
- Environment-specific parameter tuning

## Contributing

This project is designed to be easily extensible:

1. **New Environments**: Add new environment classes in `environment/`
2. **New Agents**: Extend `BaseAgent` for different RL algorithms
3. **Custom Rewards**: Modify reward functions in environment classes
4. **Network Architecture**: Adjust neural network in `DQNAgent`

