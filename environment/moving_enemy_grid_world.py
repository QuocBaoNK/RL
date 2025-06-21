import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Optional, Tuple, Dict, Any, List
from .grid_world import GridWorldEnv

class MovingEnemyGridWorldEnv(GridWorldEnv):
    """Grid World environment with moving enemies that move randomly each step"""
    
    def __init__(self, 
                 grid_size: int = 8,
                 num_rewards: int = 2,
                 num_enemies: int = 4,
                 num_obstacles: int = 8,
                 render_mode: Optional[str] = None,
                 fixed_layout: bool = True,
                 enemy_move_probability: float = 0.8,  # Xác suất enemy di chuyển mỗi step
                 use_reward_shaping: bool = True,  # Thêm tùy chọn reward shaping
                 fixed_agent_pos: Optional[Tuple[int, int]] = None,
                 fixed_rewards: Optional[List[Tuple[int, int]]] = None,
                 fixed_enemies: Optional[List[Tuple[int, int]]] = None,
                 fixed_obstacles: Optional[List[Tuple[int, int]]] = None):
        
        # Gọi constructor của class cha
        super().__init__(
            grid_size=grid_size,
            num_rewards=num_rewards,
            num_enemies=num_enemies,
            num_obstacles=num_obstacles,
            render_mode=render_mode,
            fixed_layout=fixed_layout,
            use_reward_shaping=use_reward_shaping,
            fixed_agent_pos=fixed_agent_pos,
            fixed_rewards=fixed_rewards,
            fixed_enemies=fixed_enemies,
            fixed_obstacles=fixed_obstacles
        )
        
        self.enemy_move_probability = enemy_move_probability
        
        # Cập nhật màu sắc để phân biệt với GridWorld thường
        self.colors['enemy'] = (255, 100, 100)  # Màu đỏ nhạt hơn cho moving enemy
        
    def _move_enemies(self):
        """Di chuyển enemies ngẫu nhiên"""
        new_enemies = set()
        
        for enemy_pos in list(self.enemies):
            # Kiểm tra xác suất di chuyển
            if random.random() < self.enemy_move_probability:
                # Lấy các hướng di chuyển hợp lệ
                valid_moves = []
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
                
                for dx, dy in directions:
                    new_x = enemy_pos[0] + dx
                    new_y = enemy_pos[1] + dy
                    
                    # Kiểm tra biên grid
                    if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                        new_pos = (new_x, new_y)
                        # Kiểm tra không va chạm với obstacles và không trùng với enemy khác
                        if (new_pos not in self.obstacles and 
                            new_pos not in new_enemies and
                            new_pos not in [e for e in self.enemies if e != enemy_pos]):
                            valid_moves.append(new_pos)
                
                # Thêm tùy chọn đứng yên
                if (enemy_pos not in self.obstacles and 
                    enemy_pos not in new_enemies):
                    valid_moves.append(enemy_pos)
                
                # Chọn ngẫu nhiên một hướng di chuyển hợp lệ
                if valid_moves:
                    new_pos = random.choice(valid_moves)
                    new_enemies.add(new_pos)
                else:
                    # Nếu không có hướng hợp lệ, đứng yên
                    new_enemies.add(enemy_pos)
            else:
                # Không di chuyển
                new_enemies.add(enemy_pos)
        
        self.enemies = new_enemies
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in environment với enemy di chuyển"""
        self.current_step += 1
        
        # Di chuyển agent (giống như GridWorld gốc)
        action_to_direction = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
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
        
        # Di chuyển enemies TRƯỚC khi tính reward và kiểm tra va chạm
        self._move_enemies()
        
        # Tính reward
        reward = -0.01  # Step penalty
        terminated = False
        
        # Penalty cho va chạm
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
        
        # Thêm reward shaping để giúp agent học đường đi tối ưu (tương tự GridWorld gốc)
        if self.use_reward_shaping:
            # Potential-based reward shaping: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
            current_potential = self._calculate_reward_shaping()
            gamma = 0.99  # Discount factor
            shaping_reward = gamma * current_potential - self.previous_potential
            reward += shaping_reward
            self.previous_potential = current_potential
        else:
            # Sử dụng simple distance reward (phương pháp cũ)
            distance_reward = self._calculate_distance_reward()
            reward += distance_reward
        
        # Kiểm tra thu thập reward
        if self.agent_pos in self.rewards:
            self.rewards.remove(self.agent_pos)
            self.collected_rewards += 1
            reward += 10.0
            
            # Hoàn thành mission nếu thu thập hết rewards
            if len(self.rewards) == 0:
                reward += 50.0
                terminated = True
        
        # Kiểm tra va chạm với enemy (sau khi enemy đã di chuyển)
        if self.agent_pos in self.enemies:
            reward -= 20.0
            terminated = True
        
        # Kiểm tra timeout
        if self.current_step >= self.max_steps:
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info
    
    def _render_frame(self):
        """Render one frame với visual cues cho moving enemies"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"Moving Enemy Grid World {self.grid_size}x{self.grid_size} - {'Fixed' if self.fixed_layout else 'Random'} Layout")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors['background'])
        
        # Vẽ grid
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
        
        # Vẽ rewards
        for reward_pos in self.rewards:
            pygame.draw.circle(
                canvas, self.colors['reward'],
                (reward_pos[1] * self.cell_size + self.cell_size // 2,
                 reward_pos[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )
        
        # Vẽ moving enemies với hiệu ứng đặc biệt
        for enemy_pos in self.enemies:
            # Vẽ enemy chính
            pygame.draw.rect(
                canvas, self.colors['enemy'],
                (enemy_pos[1] * self.cell_size + self.cell_size // 4,
                 enemy_pos[0] * self.cell_size + self.cell_size // 4,
                 self.cell_size // 2, self.cell_size // 2)
            )
            
            # Thêm một chấm nhỏ để biểu thị "moving"
            pygame.draw.circle(
                canvas, (255, 255, 255),  # Chấm trắng
                (enemy_pos[1] * self.cell_size + self.cell_size // 2,
                 enemy_pos[0] * self.cell_size + self.cell_size // 2),
                2
            )
        
        # Vẽ obstacles
        for obstacle_pos in self.obstacles:
            pygame.draw.rect(
                canvas, self.colors['obstacle'],
                (obstacle_pos[1] * self.cell_size,
                 obstacle_pos[0] * self.cell_size,
                 self.cell_size, self.cell_size)
            )
        
        # Vẽ agent
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
    
    def print_layout(self):
        """Print the current layout for debugging"""
        print(f"\n=== Moving Enemy Grid World Layout ({self.grid_size}x{self.grid_size}) ===")
        print(f"Layout: {'Fixed' if self.fixed_layout else 'Random'}")
        print(f"Enemy Move Probability: {self.enemy_move_probability}")
        print(f"Agent: {self.agent_pos}")
        print(f"Rewards: {list(self.rewards)}")
        print(f"Enemies: {list(self.enemies)} (moving)")
        print(f"Obstacles: {list(self.obstacles)}")
        print("=" * 50) 