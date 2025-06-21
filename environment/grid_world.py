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
                 grid_size: int = 8,  # Đổi default thành 8x8
                 num_rewards: int = 2,  # Tăng số rewards cho grid lớn hơn
                 num_enemies: int = 4,  # Tăng số enemies cho grid lớn hơn
                 num_obstacles: int = 8,  # Thêm obstacles
                 render_mode: Optional[str] = None,
                 fixed_layout: bool = True,
                 use_reward_shaping: bool = True,  # Thêm tùy chọn reward shaping
                 fixed_agent_pos: Optional[Tuple[int, int]] = None,
                 fixed_rewards: Optional[List[Tuple[int, int]]] = None,
                 fixed_enemies: Optional[List[Tuple[int, int]]] = None,
                 fixed_obstacles: Optional[List[Tuple[int, int]]] = None):
        
        self.grid_size = grid_size
        self.num_rewards = num_rewards
        self.num_enemies = num_enemies
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.fixed_layout = fixed_layout
        
        if self.fixed_layout and grid_size == 8:
            # Layout cố định cho 8x8 grid
            self.fixed_agent_pos = fixed_agent_pos or (0, 0)
            self.fixed_rewards = fixed_rewards or [(7, 7), (6, 1)]
            self.fixed_enemies = fixed_enemies or [(3, 2), (2, 5), (5, 4), (1, 6)]
            self.fixed_obstacles = fixed_obstacles or [(2, 2), (2, 3), (3, 3), (4, 2), (5, 5), (5, 6), (6, 5), (1, 4)]
        elif self.fixed_layout and grid_size == 5:
            # Giữ layout cũ cho 5x5 grid
            self.fixed_agent_pos = fixed_agent_pos or (0, 0)
            self.fixed_rewards = fixed_rewards or [(4, 4)]
            self.fixed_enemies = fixed_enemies or [(2, 1), (1, 3), (3, 2)]
            self.fixed_obstacles = fixed_obstacles or [(2, 2), (3, 1)]
        else:
            self.fixed_agent_pos = fixed_agent_pos
            self.fixed_rewards = fixed_rewards
            self.fixed_enemies = fixed_enemies
            self.fixed_obstacles = fixed_obstacles
        
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent_pos (2) + reward_vectors (num_rewards * 2) + enemy_vectors (num_enemies * 2) + obstacle_distances (8)
        obs_size = 2 + num_rewards * 2 + num_enemies * 2 + 8
        self.observation_space = spaces.Box(
            low=-grid_size, high=max(grid_size-1, 1),
            shape=(obs_size,),
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
            'enemy': (255, 0, 0),
            'obstacle': (64, 64, 64)  # Màu xám đậm cho obstacles
        }
        
        self.agent_pos = None
        self.rewards = set()
        self.enemies = set()
        self.obstacles = set()  # Thêm obstacles
        self.collected_rewards = 0
        self.max_steps = grid_size * grid_size * 3
        self.current_step = 0
        
        # Cho potential-based reward shaping
        self.previous_potential = 0.0
        self.use_reward_shaping = use_reward_shaping  # Flag để bật/tắt reward shaping
        
    def _get_obs(self) -> np.ndarray:
        """Create observation from current state using distance vectors instead of full grids"""
        # Agent position (2 dimensions)
        agent_obs = np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.float32)
        
        # Distance vectors to rewards (num_rewards * 2 dimensions)
        reward_vectors = []
        rewards_list = sorted(list(self.rewards))  # Đảm bảo thứ tự nhất quán
        for reward_pos in rewards_list:
            # Vector khoảng cách từ agent đến reward
            dx = reward_pos[0] - self.agent_pos[0]
            dy = reward_pos[1] - self.agent_pos[1]
            reward_vectors.extend([dx, dy])
        
        # Pad với zeros nếu số reward ít hơn expected
        while len(reward_vectors) < self.num_rewards * 2:
            reward_vectors.extend([0.0, 0.0])
        
        # Distance vectors to enemies (num_enemies * 2 dimensions) 
        enemy_vectors = []
        enemies_list = sorted(list(self.enemies))  # Đảm bảo thứ tự nhất quán
        for enemy_pos in enemies_list:
            # Vector khoảng cách từ agent đến enemy
            dx = enemy_pos[0] - self.agent_pos[0]
            dy = enemy_pos[1] - self.agent_pos[1]
            enemy_vectors.extend([dx, dy])
        
        # Pad với zeros nếu số enemy ít hơn expected
        while len(enemy_vectors) < self.num_enemies * 2:
            enemy_vectors.extend([0.0, 0.0])
        
        # Distance to nearest obstacles in 8 directions (8 dimensions)
        obstacle_distances = self._get_obstacle_distances_8_directions()
        
        reward_vectors = np.array(reward_vectors, dtype=np.float32)
        enemy_vectors = np.array(enemy_vectors, dtype=np.float32)
        obstacle_distances = np.array(obstacle_distances, dtype=np.float32)
        
        return np.concatenate([agent_obs, reward_vectors, enemy_vectors, obstacle_distances])
    
    def _get_obstacle_distances_8_directions(self) -> List[float]:
        """Get distance to nearest obstacle/wall in 8 directions from agent position"""
        directions = [
            (-1, 0),   # UP
            (-1, 1),   # UP-RIGHT  
            (0, 1),    # RIGHT
            (1, 1),    # DOWN-RIGHT
            (1, 0),    # DOWN
            (1, -1),   # DOWN-LEFT
            (0, -1),   # LEFT
            (-1, -1)   # UP-LEFT
        ]
        
        distances = []
        agent_x, agent_y = self.agent_pos
        
        for dx, dy in directions:
            distance = 1.0  # Start from distance 1
            current_x, current_y = agent_x + dx, agent_y + dy
            
            # Check each step in this direction until hitting obstacle or wall
            while (0 <= current_x < self.grid_size and 
                   0 <= current_y < self.grid_size and
                   (current_x, current_y) not in self.obstacles):
                distance += 1.0
                current_x += dx
                current_y += dy
            
            # Normalize distance by grid size for better learning
            normalized_distance = distance / self.grid_size
            distances.append(normalized_distance)
        
        return distances
    
    def _calculate_reward_shaping(self) -> float:
        """
        Tính reward shaping dựa trên khoảng cách đến rewards - càng gần reward càng cao
        Sử dụng potential-based reward shaping để đảm bảo optimal policy không đổi
        """
        if not self.rewards:
            return 0.0
        
        # Tìm khoảng cách nhỏ nhất đến reward gần nhất
        min_distance_to_reward = float('inf')
        for reward_pos in self.rewards:
            # Sử dụng Manhattan distance
            distance = abs(self.agent_pos[0] - reward_pos[0]) + abs(self.agent_pos[1] - reward_pos[1])
            min_distance_to_reward = min(min_distance_to_reward, distance)
        
        # Potential-based reward shaping: Φ(s) = -distance_to_nearest_reward
        # Reward shaping: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
        # Ở đây ta chỉ trả về potential function để tính trong step()
        max_possible_distance = (self.grid_size - 1) * 2  # Max Manhattan distance
        
        # Normalize và scale potential function
        potential = -(min_distance_to_reward / max_possible_distance)
        
        return potential
    
    def _calculate_distance_reward(self) -> float:
        """
        Tính immediate reward dựa trên việc tiến gần hơn đến reward
        Đây là simplified version, không dùng potential-based shaping
        """
        if not self.rewards:
            return 0.0
        
        # Tìm khoảng cách nhỏ nhất đến reward gần nhất
        min_distance_to_reward = float('inf')
        for reward_pos in self.rewards:
            distance = abs(self.agent_pos[0] - reward_pos[0]) + abs(self.agent_pos[1] - reward_pos[1])
            min_distance_to_reward = min(min_distance_to_reward, distance)
        
        # Reward cao hơn khi gần reward (khoảng cách nhỏ)
        max_possible_distance = (self.grid_size - 1) * 2
        
        # Tạo reward gradient hướng về phần thưởng
        # Sử dụng exponential decay để tạo gradient mạnh hơn khi gần
        normalized_distance = min_distance_to_reward / max_possible_distance
        distance_reward = 0.5 * (1.0 - normalized_distance)  # 0.0 to 0.5 range
        
        return distance_reward
    
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
            
            self.obstacles = set(self.fixed_obstacles) if self.fixed_obstacles else set()
            if len(self.obstacles) != self.num_obstacles:
                print(f"Warning: Fixed obstacles ({len(self.obstacles)}) != num_obstacles ({self.num_obstacles})")
        else:
            self.agent_pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            
            # Tạo obstacles trước để tránh xung đột
            self.obstacles = set()
            while len(self.obstacles) < self.num_obstacles:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                )
                if pos != self.agent_pos:
                    self.obstacles.add(pos)
            
            self.rewards = set()
            while len(self.rewards) < self.num_rewards:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                )
                if pos != self.agent_pos and pos not in self.obstacles:
                    self.rewards.add(pos)
            
            self.enemies = set()
            while len(self.enemies) < self.num_enemies:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                )
                if pos != self.agent_pos and pos not in self.rewards and pos not in self.obstacles:
                    self.enemies.add(pos)
        
        # Tính potential ban đầu cho reward shaping
        if self.use_reward_shaping:
            self.previous_potential = self._calculate_reward_shaping()
        
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
        
        # Kiểm tra va chạm với obstacles
        if new_pos in self.obstacles:
            new_pos = old_pos  # Không di chuyển nếu va chạm với obstacle
        
        self.agent_pos = new_pos
        
        reward = -0.01
        terminated = False
        
        if old_pos == new_pos:
            # Kiểm tra lý do không di chuyển được
            attempted_pos = (
                max(0, min(self.grid_size - 1, old_pos[0] + direction[0])),
                max(0, min(self.grid_size - 1, old_pos[1] + direction[1]))
            )
            if attempted_pos in self.obstacles:
                reward -= 0.1  # Penalty cao hơn cho va chạm với obstacle
            else:
                reward -= 0.05  # Penalty thông thường cho va chạm với tường
        
        # Thêm reward shaping để giúp agent học đường đi tối ưu
        if self.use_reward_shaping:
            # Potential-based reward shaping: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
            current_potential = self._calculate_reward_shaping()
            gamma = 0.99  # Discount factor, có thể truyền từ tham số
            shaping_reward = gamma * current_potential - self.previous_potential
            reward += shaping_reward
            self.previous_potential = current_potential
        else:
            # Sử dụng simple distance reward (phương pháp cũ)
            distance_reward = self._calculate_distance_reward()
            reward += distance_reward
        
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
        
        # Vẽ obstacles
        for obstacle_pos in self.obstacles:
            pygame.draw.rect(
                canvas, self.colors['obstacle'],
                (obstacle_pos[1] * self.cell_size,
                 obstacle_pos[0] * self.cell_size,
                 self.cell_size, self.cell_size)
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
        print(f"Obstacles: {list(self.obstacles)}")
        print("=" * 40) 