import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Optional, Tuple, Dict, Any, List

class GridWorldEnv(gym.Env):
    """Grid World 2D environment with agent, rewards and enemies"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 grid_size: int = 5,
                 num_rewards: int = 1,
                 num_enemies: int = 3,
                 render_mode: Optional[str] = None,
                 fixed_layout: bool = True,
                 fixed_agent_pos: Optional[Tuple[int, int]] = None,
                 fixed_rewards: Optional[List[Tuple[int, int]]] = None,
                 fixed_enemies: Optional[List[Tuple[int, int]]] = None):
        
        self.grid_size = grid_size
        self.num_rewards = num_rewards
        self.num_enemies = num_enemies
        self.render_mode = render_mode
        self.fixed_layout = fixed_layout
        
        if self.fixed_layout and grid_size == 5:
            self.fixed_agent_pos = fixed_agent_pos or (0, 0)
            self.fixed_rewards = fixed_rewards or [(4, 4)]
            self.fixed_enemies = fixed_enemies or [(2, 1), (1, 3), (3, 2)]
        else:
            self.fixed_agent_pos = fixed_agent_pos
            self.fixed_rewards = fixed_rewards
            self.fixed_enemies = fixed_enemies
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size-1, 1),
            shape=(2 + 2 * grid_size * grid_size,),
            dtype=np.float32
        )
        
        self.window_size = 400
        self.cell_size = self.window_size // grid_size
        self.window = None
        self.clock = None
        
        self.colors = {
            'background': (255, 255, 255),
            'grid': (128, 128, 128),
            'agent': (0, 0, 255),
            'reward': (0, 255, 0),
            'enemy': (255, 0, 0)
        }
        
        self.agent_pos = None
        self.rewards = set()
        self.enemies = set()
        self.collected_rewards = 0
        self.max_steps = grid_size * grid_size * 3
        self.current_step = 0
        
    def _get_obs(self) -> np.ndarray:
        """Create observation from current state"""
        agent_obs = np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.float32)
        
        rewards_grid = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        for reward_pos in self.rewards:
            idx = reward_pos[0] * self.grid_size + reward_pos[1]
            rewards_grid[idx] = 1.0
            
        enemies_grid = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        for enemy_pos in self.enemies:
            idx = enemy_pos[0] * self.grid_size + enemy_pos[1]
            enemies_grid[idx] = 1.0
            
        return np.concatenate([agent_obs, rewards_grid, enemies_grid])
    
    def _get_info(self) -> Dict[str, Any]:
        """Additional environment information"""
        return {
            "collected_rewards": self.collected_rewards,
            "remaining_rewards": len(self.rewards),
            "current_step": self.current_step,
            "agent_position": self.agent_pos,
            "layout_type": "fixed" if self.fixed_layout else "random"
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.collected_rewards = 0
        
        if self.fixed_layout and self.fixed_agent_pos:
            self.agent_pos = self.fixed_agent_pos
            
            self.rewards = set(self.fixed_rewards) if self.fixed_rewards else set()
            if len(self.rewards) != self.num_rewards:
                print(f"Warning: Fixed rewards ({len(self.rewards)}) != num_rewards ({self.num_rewards})")
            
            self.enemies = set(self.fixed_enemies) if self.fixed_enemies else set()
            if len(self.enemies) != self.num_enemies:
                print(f"Warning: Fixed enemies ({len(self.enemies)}) != num_enemies ({self.num_enemies})")
        else:
            self.agent_pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            
            self.rewards = set()
            while len(self.rewards) < self.num_rewards:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                )
                if pos != self.agent_pos:
                    self.rewards.add(pos)
            
            self.enemies = set()
            while len(self.enemies) < self.num_enemies:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                )
                if pos != self.agent_pos and pos not in self.rewards:
                    self.enemies.add(pos)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in environment"""
        self.current_step += 1
        
        action_to_direction = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        
        old_pos = self.agent_pos
        direction = action_to_direction[action]
        new_pos = (
            max(0, min(self.grid_size - 1, self.agent_pos[0] + direction[0])),
            max(0, min(self.grid_size - 1, self.agent_pos[1] + direction[1]))
        )
        self.agent_pos = new_pos
        
        reward = -0.01
        terminated = False
        
        if old_pos == new_pos:
            reward -= 0.05
        
        if self.agent_pos in self.rewards:
            self.rewards.remove(self.agent_pos)
            self.collected_rewards += 1
            reward += 10.0
            
            if len(self.rewards) == 0:
                reward += 50.0
                terminated = True
        
        if self.agent_pos in self.enemies:
            reward -= 20.0
            terminated = True
        
        if self.current_step >= self.max_steps:
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render one frame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"Grid World {self.grid_size}x{self.grid_size} - {'Fixed' if self.fixed_layout else 'Random'} Layout")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors['background'])
        
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size), 1
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, self.colors['grid'],
                (0, y * self.cell_size),
                (self.window_size, y * self.cell_size), 1
            )
        
        for reward_pos in self.rewards:
            pygame.draw.circle(
                canvas, self.colors['reward'],
                (reward_pos[1] * self.cell_size + self.cell_size // 2,
                 reward_pos[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )
        
        for enemy_pos in self.enemies:
            pygame.draw.rect(
                canvas, self.colors['enemy'],
                (enemy_pos[1] * self.cell_size + self.cell_size // 4,
                 enemy_pos[0] * self.cell_size + self.cell_size // 4,
                 self.cell_size // 2, self.cell_size // 2)
            )
        
        pygame.draw.circle(
            canvas, self.colors['agent'],
            (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
             self.agent_pos[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 2 - 2
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def print_layout(self):
        """Print the current layout for debugging"""
        print(f"\n=== Grid World Layout ({self.grid_size}x{self.grid_size}) ===")
        print(f"Layout: {'Fixed' if self.fixed_layout else 'Random'}")
        print(f"Agent: {self.agent_pos}")
        print(f"Rewards: {list(self.rewards)}")
        print(f"Enemies: {list(self.enemies)}")
        print("=" * 40) 