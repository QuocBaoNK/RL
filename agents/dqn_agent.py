import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from .base_agent import BaseAgent

class DQNNetwork(nn.Module):
    """Deep Q-Network with configurable architecture for different environments"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64], output_size: int = 4):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DQNAgent(BaseAgent):
    """Deep Q-Network Agent for various environments"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int = 4,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 100,
                 hidden_sizes: List[int] = [128, 64]):
        
        super().__init__(action_size, lr, gamma, epsilon, epsilon_min, epsilon_decay)
        
        self.state_size = state_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.hidden_sizes = hidden_sizes
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create networks
        self.q_network = DQNNetwork(
            state_size, 
            hidden_sizes=hidden_sizes, 
            output_size=action_size
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_size, 
            hidden_sizes=hidden_sizes, 
            output_size=action_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.update_target_network()
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.step_count = 0
        self.losses = []
        
    def update_target_network(self):
        """Copy weights from main to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """Update the agent using experience replay"""
        self.remember(state, action, reward, next_state, done)
        
        loss = self.replay()
        
        return loss
    
    def replay(self) -> Optional[float]:
        """Train the model on a batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_epsilon()
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        return loss_value
    
    def save(self, filepath: str):
        """Save model with complete state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'hidden_sizes': self.hidden_sizes
            },
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'losses': self.losses,
                'training_time': self.training_time
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load model with complete state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.step_count = checkpoint.get('step_count', 0)
        
        # Load config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.state_size = config.get('state_size', self.state_size)
            self.action_size = config.get('action_size', self.action_size)
            self.lr = config.get('lr', self.lr)
            self.gamma = config.get('gamma', self.gamma)
            self.epsilon_min = config.get('epsilon_min', self.epsilon_min)
            self.epsilon_decay = config.get('epsilon_decay', self.epsilon_decay)
            self.hidden_sizes = config.get('hidden_sizes', self.hidden_sizes)
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.episode_rewards = metrics.get('episode_rewards', [])
            self.episode_lengths = metrics.get('episode_lengths', [])
            self.losses = metrics.get('losses', [])
            self.training_time = metrics.get('training_time', 0)
        
        self.q_network.eval()
        print(f"Model loaded from {filepath}")
        print(f"  State size: {self.state_size}")
        print(f"  Action size: {self.action_size}")
        print(f"  Architecture: {self.hidden_sizes}")
        print(f"  Training episodes: {len(self.episode_rewards)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'step_count': self.step_count
        }
    
    def _plot_agent_specific(self, ax1, ax2):
        """Plot DQN specific metrics"""
        # Training loss
        if self.losses:
            ax1.plot(self.losses, alpha=0.6, label='Loss')
            if len(self.losses) > 50:
                window = min(50, len(self.losses) // 10)
                moving_avg = np.convolve(self.losses, 
                                       np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.losses)), 
                        moving_avg, 'r-', linewidth=2, label='Moving Avg')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
        
        # Epsilon decay
        epsilons = []
        for i in range(len(self.episode_rewards)):
            epsilon = max(self.epsilon_min, 
                         1.0 * (self.epsilon_decay ** i))
            epsilons.append(epsilon)
        
        if epsilons:
            ax2.plot(epsilons, 'g-', linewidth=2)
            ax2.set_title('Epsilon Decay')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.grid(True)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the neural network"""
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        return {
            'architecture': self.hidden_sizes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_size': self.state_size,
            'output_size': self.action_size
        } 