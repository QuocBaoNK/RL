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
    """Deep Q-Network with improved architecture"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 4):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    """DQN Agent for Grid World"""
    
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
                 target_update: int = 100):
        
        super().__init__(action_size, lr, gamma, epsilon, epsilon_min, epsilon_decay)
        
        self.state_size = state_size
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQNNetwork(state_size, hidden_size=128, output_size=action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size=128, output_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()
        
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
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'losses': self.losses
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.step_count = checkpoint.get('step_count', 0)
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.episode_rewards = metrics.get('episode_rewards', [])
            self.episode_lengths = metrics.get('episode_lengths', [])
            self.losses = metrics.get('losses', [])
        
        self.q_network.eval()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0
        }
    
    def _plot_agent_specific(self, ax1, ax2):
        """Plot DQN specific metrics"""
        if self.losses:
            ax1.plot(self.losses, alpha=0.6)
            if len(self.losses) > 50:
                window = min(50, len(self.losses) // 10)
                moving_avg = np.convolve(self.losses, 
                                       np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.losses)), 
                               moving_avg, 'r-', linewidth=2)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
        
        epsilons = []
        for i in range(len(self.episode_rewards)):
            epsilon = max(self.epsilon_min, 
                         1.0 * (self.epsilon_decay ** i))
            epsilons.append(epsilon)
        
        ax2.plot(epsilons)
        ax2.set_title('Epsilon Decay')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True) 