import numpy as np
import matplotlib.pyplot as plt
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from environment import GridWorldEnv


class BaseAgent(ABC):
    """Base class for all RL agents with integrated training functionality"""
    
    def __init__(self,
                 action_size: int = 4,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_time = 0
        
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action given the current state"""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """Update the agent based on experience"""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save the agent's model/parameters"""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load the agent's model/parameters"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics"""
        pass
    
    def update_epsilon(self):
        """Update epsilon for exploration-exploitation balance"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self,
              env_config: dict = None,
              episodes: int = 1000,
              max_steps_per_episode: int = 75,
              save_dir: str = "models",
              save_interval: int = 100,
              print_interval: int = 50) -> Tuple[List[float], List[int]]:
        """Train the agent in the environment"""
        
        if env_config is None:
            env_config = {
                "grid_size": 5,
                "num_rewards": 1,
                "num_enemies": 3,
                "render_mode": None,
                "fixed_layout": True
            }
        
        env = GridWorldEnv(**env_config)
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training {self.__class__.__name__} for {episodes} episodes...")
        print(f"Environment: {env_config['grid_size']}x{env_config['grid_size']} {'Fixed' if env_config.get('fixed_layout', False) else 'Random'} grid")
        print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                action = self.act(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                self.update(state, action, reward, next_state, terminated)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if (episode + 1) % print_interval == 0:
                self._print_progress(episode + 1, episodes, print_interval, time.time() - start_time)
            
            if (episode + 1) % save_interval == 0:
                filename = f"{self.__class__.__name__.lower()}_episode_{episode + 1}.pkl"
                self.save(os.path.join(save_dir, filename))
        
        self.training_time = time.time() - start_time
        print(f"\nTraining completed! Time: {self.training_time:.1f}s")
        
        final_filename = f"{self.__class__.__name__.lower()}_final.pkl"
        if self.__class__.__name__ == "DQNAgent":
            final_filename = final_filename.replace('.pkl', '.pth')
        
        self.save(os.path.join(save_dir, final_filename))
        print(f"Model saved: {os.path.join(save_dir, final_filename)}")
        
        env.close()
        return self.episode_rewards, self.episode_lengths
    
    def evaluate(self,
                 env_config: dict = None,
                 episodes: int = 10,
                 render: bool = True,
                 max_steps: int = 75) -> Tuple[float, float]:
        """Evaluate the trained agent"""
        
        if env_config is None:
            env_config = {
                "grid_size": 5,
                "num_rewards": 1,
                "num_enemies": 3,
                "render_mode": "human" if render else None,
                "fixed_layout": True
            }
        
        env = GridWorldEnv(**env_config)
        total_rewards = []
        successes = 0
        
        print(f"Evaluating {self.__class__.__name__} for {episodes} episodes...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action = self.act(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    time.sleep(0.1)
                
                if terminated or truncated:
                    if info.get('remaining_rewards', 1) == 0:
                        successes += 1
                    break
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
        
        avg_reward = np.mean(total_rewards)
        success_rate = successes / episodes
        
        print(f"\nResults:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")
        
        env.close()
        return avg_reward, success_rate
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) > 50:
            window = min(50, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                           moving_avg, 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.episode_lengths, alpha=0.6)
        if len(self.episode_lengths) > 50:
            window = min(50, len(self.episode_lengths) // 10)
            moving_avg = np.convolve(self.episode_lengths, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.episode_lengths)), 
                           moving_avg, 'r-', linewidth=2)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        self._plot_agent_specific(axes[1, 0], axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_progress(self, episode: int, total_episodes: int, 
                       print_interval: int, elapsed_time: float):
        """Print training progress"""
        avg_reward = np.mean(self.episode_rewards[-print_interval:])
        avg_length = np.mean(self.episode_lengths[-print_interval:])
        stats = self.get_stats()
        
        print(f"Episode {episode}/{total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Length: {avg_length:.1f}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        
        for key, value in stats.items():
            if key not in ['epsilon']:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"  Time: {elapsed_time:.1f}s")
        print("-" * 30)
    
    @abstractmethod
    def _plot_agent_specific(self, ax1, ax2):
        """Plot agent-specific metrics (to be implemented by subclasses)"""
        pass 