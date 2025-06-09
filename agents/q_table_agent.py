import numpy as np
import random
import pickle
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class QTableAgent(BaseAgent):
    """Q-Table Agent using Tabular Q-Learning"""
    
    def __init__(self,
                 action_size: int = 4,
                 lr: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        super().__init__(action_size, lr, gamma, epsilon, epsilon_min, epsilon_decay)
        
        self.q_table = {}
        
        self.q_table_sizes = []
        self.avg_q_values = []
        
    def _get_state_key(self, observation: np.ndarray) -> str:
        """Convert observation to state key for Q-table"""
        agent_x, agent_y = int(observation[0]), int(observation[1])
        
        remaining_obs = len(observation) - 2
        grid_size = int(np.sqrt(remaining_obs / 2))
        
        rewards_grid = observation[2:2 + grid_size * grid_size]
        rewards_positions = tuple(np.where(rewards_grid == 1.0)[0])
        
        enemies_grid = observation[2 + grid_size * grid_size:]
        enemies_positions = tuple(np.where(enemies_grid == 1.0)[0])
        
        state_key = f"{agent_x},{agent_y}|R{rewards_positions}|E{enemies_positions}"
        return state_key
    
    def _init_state(self, state_key: str):
        """Initialize Q-values for a new state"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in range(self.action_size)}
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        state_key = self._get_state_key(state)
        self._init_state(state_key)
        
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_q_values = self.q_table[state_key]
        return max(state_q_values, key=state_q_values.get)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """Update Q-table using Q-learning update rule"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        self._init_state(state_key)
        self._init_state(next_state_key)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_table[next_state_key].values())
            target_q = reward + self.gamma * max_next_q
        
        self.q_table[state_key][action] = current_q + self.lr * (target_q - current_q)
        
        self.update_epsilon()
        
        self._update_metrics()
        
        return abs(target_q - current_q)
    
    def save(self, filepath: str):
        """Save Q-table to file"""
        save_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'config': {
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            },
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'q_table_sizes': self.q_table_sizes,
                'avg_q_values': self.avg_q_values
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.epsilon = save_data.get('epsilon', 0.01)
        
        if 'config' in save_data:
            config = save_data['config']
            self.action_size = config['action_size']
            self.lr = config['lr']
            self.gamma = config['gamma']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_decay = config['epsilon_decay']
        
        if 'metrics' in save_data:
            metrics = save_data['metrics']
            self.episode_rewards = metrics.get('episode_rewards', [])
            self.episode_lengths = metrics.get('episode_lengths', [])
            self.q_table_sizes = metrics.get('q_table_sizes', [])
            self.avg_q_values = metrics.get('avg_q_values', [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'q_table_size': len(self.q_table),
            'avg_q_value': self.avg_q_values[-1] if self.avg_q_values else 0.0
        }
    
    def _update_metrics(self):
        """Update Q-Table specific metrics"""
        if len(self.episode_rewards) > len(self.q_table_sizes):
            self.q_table_sizes.append(len(self.q_table))
            
            if self.q_table:
                all_q_values = []
                for state_q in self.q_table.values():
                    all_q_values.extend(state_q.values())
                self.avg_q_values.append(np.mean(all_q_values))
            else:
                self.avg_q_values.append(0.0)
    
    def _plot_agent_specific(self, ax1, ax2):
        """Plot Q-Table specific metrics"""
        if self.q_table_sizes:
            ax1.plot(self.q_table_sizes)
            ax1.set_title('Q-Table Size Growth')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Number of States')
            ax1.grid(True)
        
        if self.avg_q_values:
            ax2.plot(self.avg_q_values)
            if len(self.avg_q_values) > 50:
                window = min(50, len(self.avg_q_values) // 10)
                moving_avg = np.convolve(self.avg_q_values, 
                                       np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.avg_q_values)), 
                        moving_avg, 'r-', linewidth=2)
            ax2.set_title('Average Q-Values')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Q-Value')
            ax2.grid(True)
    
    def print_q_table_sample(self, num_states: int = 5):
        """Print a sample of the Q-table for debugging"""
        print(f"\n=== Q-Table Sample (showing {min(num_states, len(self.q_table))} states) ===")
        print(f"Total states in Q-table: {len(self.q_table)}")
        
        if not self.q_table:
            print("Q-table is empty!")
            return
        
        states = list(self.q_table.keys())[:num_states]
        action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        
        for i, state in enumerate(states):
            q_values = self.q_table[state]
            best_action = max(q_values, key=q_values.get)
            print(f"\nState {i+1}: {state}")
            for action, q_value in q_values.items():
                action_name = action_names.get(action, str(action))
                print(f"  {action_name}: {q_value:.3f}")
            print(f"  Best action: {action_names.get(best_action, str(best_action))} ({q_values[best_action]:.3f})")
            print("-" * 40)
        
        all_q_values = []
        for state_q in self.q_table.values():
            all_q_values.extend(state_q.values())
        
        if all_q_values:
            print(f"\nQ-Table Statistics:")
            print(f"  Mean Q-value: {np.mean(all_q_values):.3f}")
            print(f"  Max Q-value: {np.max(all_q_values):.3f}")
            print(f"  Min Q-value: {np.min(all_q_values):.3f}")
            print(f"  Std Q-value: {np.std(all_q_values):.3f}")
        print("=" * 50) 