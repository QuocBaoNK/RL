import numpy as np
import matplotlib.pyplot as plt
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from environment import GridWorldEnv


class BaseAgent(ABC):
    """Base class for all Deep RL agents with integrated training functionality"""
    
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
              max_steps_per_episode: int = 192,
              save_dir: str = "models",
              save_interval: int = 100,
              print_interval: int = 50) -> Tuple[List[float], List[int]]:
        """
        Train the agent using the provided environment configuration
        
        Args:
            env_config: Dictionary containing environment configuration
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_dir: Directory to save models
            save_interval: Save model every N episodes
            print_interval: Print progress every N episodes
        
        Returns:
            Tuple of (episode_rewards, episode_lengths)
        """
        print(f"🚀 Starting training for {episodes} episodes...")
        
        # Create environment using the config
        env = self._create_environment(env_config)
        
        # Reset tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = time.time()
        
        for episode in range(episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.act(state, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update agent
                loss = self.update(state, action, reward, next_state, done)
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Update epsilon
            self.update_epsilon()
            
            # Store episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                avg_length = np.mean(self.episode_lengths[-print_interval:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = f"{save_dir}/{env_config.get('env_type', 'unknown')}_dqnagent_ep{episode + 1}.pth"
                self.save(model_path)
                print(f"💾 Model saved: {model_path}")
        
        # Record training time
        self.training_time = time.time() - start_time
        
        print(f"✅ Training completed in {self.training_time:.2f} seconds")
        print(f"📊 Final Stats - Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        env.close()
        return self.episode_rewards, self.episode_lengths
    
    def _create_environment(self, env_config: dict):
        """Create environment based on config - can be overridden for different env types"""
        from environment import GridWorldEnv, MovingEnemyGridWorldEnv
        
        # Remove env_type from config before passing to environment constructor
        env_config_copy = env_config.copy()
        env_type = env_config_copy.pop('env_type', 'gridworld')
        
        if env_type == 'gridworld':
            return GridWorldEnv(**env_config_copy)
        elif env_type == 'movingenemyworld':
            return MovingEnemyGridWorldEnv(**env_config_copy)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def evaluate(self,
                 env_config: dict = None,
                 episodes: int = 10,
                 render: bool = True,
                 max_steps: int = 192) -> Tuple[float, float]:
        """Evaluate the trained agent"""
        
        if env_config is None:
            env_config = {
                "grid_size": 8,
                "num_rewards": 2,
                "num_enemies": 4,
                "num_obstacles": 8,
                "render_mode": "human" if render else None,
                "fixed_layout": True
            }
        
        # Create environment for evaluation
        eval_env_config = env_config.copy()
        if render:
            eval_env_config["render_mode"] = "human"
        else:
            eval_env_config["render_mode"] = None
            
        env = self._create_environment(eval_env_config)
        
        total_rewards = []
        successful_episodes = 0
        
        print(f"🧪 Evaluating agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state, info = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Use greedy policy (no exploration)
                action = self.act(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if render:
                    env.render()
                    time.sleep(0.1)  # Small delay for visualization
                
                if terminated or truncated:
                    if info.get('collected_rewards', 0) == env.num_rewards:
                        successful_episodes += 1
                    break
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Success = {'✅' if info.get('collected_rewards', 0) == env.num_rewards else '❌'}")
        
        env.close()
        
        avg_reward = np.mean(total_rewards)
        success_rate = successful_episodes / episodes
        
        return avg_reward, success_rate
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if not self.episode_rewards:
            print("No training data to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards, alpha=0.7, label='Episode Reward')
        
        # Plot moving average
        window_size = min(100, len(self.episode_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(self.episode_rewards)), 
                    moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths, alpha=0.7, color='green', label='Episode Length')
        
        # Plot moving average for lengths
        if window_size > 1:
            moving_avg_length = np.convolve(self.episode_lengths, 
                                          np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(self.episode_lengths)), 
                    moving_avg_length, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plot saved: {save_path}")
        
        plt.show() 