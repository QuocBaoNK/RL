# Grid World Reinforcement Learning

Simplified Grid World environment for training and testing reinforcement learning agents. Features a clean object-oriented architecture with training functionality integrated directly into agent classes.

## Architecture Highlights

- **Base Agent Class**: Abstract base class providing common training, evaluation, and plotting functionality
- **Integrated Training**: Each agent contains its own training logic, eliminating the need for separate trainer classes
- **Unified Interface**: Both DQN and Q-Table agents share the same API for training and evaluation
- **Minimal Code Duplication**: Common functionality is inherited from BaseAgent

## Features

- **5x5 Fixed Layout**: Default environment with strategic placement of rewards and enemies
- **Two Agent Types**: DQN (neural network) and Q-Table (tabular) implementations
- **Clean CLI Interface**: Command-line tool with comprehensive argument parsing
- **Training & Evaluation**: Built-in training pipelines with visualization and metrics
- **Human Play Mode**: Interactive mode for manual testing and exploration

## Default Environment

The default configuration uses a **5x5 grid** with a **fixed layout**:
- **Agent Start**: Top-left corner (0,0)
- **1 Reward**: Bottom-right corner (4,4) - goal to reach
- **3 Enemies**: Strategic positions at (2,1), (1,3), (3,2) - obstacles to avoid

This configuration provides:
- **Clear objective**: Single reward makes the goal obvious
- **Strategic challenge**: Multiple enemies require careful navigation
- **Consistent training**: Fixed layout ensures reproducible results
- **Balanced difficulty**: Manageable for both learning algorithms

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Train Q-Table Agent (Recommended for 5x5)
```bash
python main.py train --agent-type qtable --episodes 1000
```

#### Train DQN Agent
```bash
python main.py train --agent-type dqn --episodes 1000
```

#### Test Trained Agent
```bash
python main.py test --model models/qtableagent_final.pkl --render
```

#### Human Play Mode
```bash
python main.py play
```

#### Compare Agents
```bash
python main.py compare
```

## Architecture Overview

### Class Hierarchy
```
BaseAgent (abstract)
├── QTableAgent
└── DQNAgent
```

### BaseAgent Methods
- `train()`: Complete training pipeline with environment interaction
- `evaluate()`: Test agent performance with metrics
- `plot_training_progress()`: Visualize training metrics
- `act()`: Choose action (abstract - implemented by subclasses)
- `update()`: Update agent based on experience (abstract)
- `save()/load()`: Model persistence (abstract)

### Agent-Specific Features

**QTableAgent**:
- Dictionary-based Q-value storage
- State representation as string keys
- Q-Table size tracking and visualization
- Direct Q-learning updates

**DQNAgent**:
- Neural network function approximation
- Experience replay buffer
- Target network for stability
- Gradient-based updates

## Command Line Interface

### Available Commands

- `play` - Human control mode (W/S/A/D keys)
- `train` - Train RL agent (DQN or Q-Table)
- `test` - Test trained agent
- `compare` - Compare DQN vs Q-Table performance

### Common Arguments

- `--grid-size` - Grid size (default: 5)
- `--num-rewards` - Number of rewards (default: 1)
- `--num-enemies` - Number of enemies (default: 3)
- `--fixed-layout` - Use fixed layout (default: True)
- `--random-layout` - Use random layout instead
- `--max-steps` - Max steps per episode (default: 75)

### Training Arguments

- `--agent-type {dqn,qtable}` - Agent type (default: qtable)
- `--episodes` - Training episodes (default: 1000)
- `--learning-rate` - Learning rate (0.1 for Q-Table, 0.001 for DQN)
- `--epsilon` - Initial exploration rate (default: 1.0)
- `--gamma` - Discount factor (default: 0.99)
- `--plot` - Generate training plots
- `--evaluate` - Evaluate after training

## Agent Comparison

| Feature | Q-Table | DQN |
|---------|---------|-----|
| **Type** | Tabular | Neural Network |
| **Memory** | Dictionary | Experience Replay |
| **Learning Rate** | 0.1 | 0.001 |
| **Best for** | Small grids (≤6x6) | Large grids (>6x6) |
| **Training Speed** | Fast | Moderate |
| **Convergence** | Quick on simple tasks | Stable on complex tasks |
| **File Format** | .pkl | .pth |

## Code Example

```python
from agents import QTableAgent, DQNAgent
from environment import GridWorldEnv

# Create and train Q-Table agent
agent = QTableAgent(lr=0.1, gamma=0.99)
rewards, lengths = agent.train(episodes=1000)
agent.plot_training_progress("qtable_progress.png")

# Evaluate trained agent
avg_reward, success_rate = agent.evaluate(episodes=10, render=True)

# Save/Load
agent.save("models/my_qtable.pkl")
agent.load("models/my_qtable.pkl")
```

## File Structure

```
RL/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Abstract base class with training logic
│   ├── dqn_agent.py       # Deep Q-Network agent
│   ├── q_table_agent.py   # Q-Table agent
│   └── __init__.py
├── environment/            # Environment implementation
│   ├── grid_world.py      # Grid World environment
│   └── __init__.py
├── training/               # (Deprecated - functionality moved to agents)
│   └── __init__.py
├── models/                 # Saved models
│   ├── *.pth             # DQN models
│   └── *.pkl             # Q-Table models
├── main.py                # Main CLI interface
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Performance Tips

### For Q-Table Agents:
- Use smaller grids (5x5 to 6x6) for optimal performance
- Higher learning rate (0.1) works well
- Fast convergence on simple layouts
- Watch Q-table size growth during training

### For DQN Agents:
- Better for larger grids (7x7+) or complex environments
- Lower learning rate (0.001) prevents instability
- Requires more episodes but handles complexity better
- Monitor loss curves and epsilon decay

### Environment Recommendations:
- **Fixed layout**: Use for consistent training and comparison
- **Random layout**: Use for generalization and robustness testing
- **Single reward**: Clearer learning objective for beginners
- **Multiple rewards**: More complex strategies and longer episodes

## Extending the Framework

To add a new agent type:

1. Create a new class inheriting from `BaseAgent`
2. Implement abstract methods: `act()`, `update()`, `save()`, `load()`, `get_stats()`, `_plot_agent_specific()`
3. Add agent-specific logic and parameters
4. Import in `agents/__init__.py`
5. Update `main.py` to support the new agent type

Example:
```python
from agents.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def act(self, state, training=True):
        # Implementation
        pass
    
    def update(self, state, action, reward, next_state, done):
        # SARSA update logic
        pass
    # ... other required methods
```

## Troubleshooting

**Training not converging?**
- Try adjusting learning rate
- Increase training episodes
- Check epsilon decay settings
- Use fixed layout for debugging

**Q-Table too large?**
- Reduce grid size
- Fewer rewards/enemies
- Consider DQN for larger spaces

**DQN unstable?**
- Lower learning rate
- Increase replay buffer size
- Adjust target network update frequency
- More training episodes

## License

MIT License - feel free to use and modify for your projects. 