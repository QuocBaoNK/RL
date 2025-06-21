#!/usr/bin/env python3
"""
Multi-Environment Deep RL Framework
Supports multiple environment types with corresponding DQN agents
Each environment has its own specialized DQN agent with custom input processing
"""

import argparse
import os
import sys
import pygame
import time
from typing import Optional, Dict, Any

from environment import GridWorldEnv, MovingEnemyGridWorldEnv
from agents import DQNAgent


def play_human(args):
    """Human control mode for any environment"""
    print("🎮 Human Control Mode")
    print("Controls: W/S/A/D or Arrow Keys, ESC to quit")
    
    # Create environment based on type
    env = create_environment(args)
    
    state, info = env.reset()
    env.print_layout()
    
    pygame.init()
    
    action_map = {
        pygame.K_UP: 0, pygame.K_w: 0,
        pygame.K_RIGHT: 1, pygame.K_d: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2,
        pygame.K_LEFT: 3, pygame.K_a: 3,
    }
    
    running = True
    total_reward = 0
    steps = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset environment
                    state, info = env.reset()
                    total_reward = 0
                    steps = 0
                    env.print_layout()
                    print(f"Environment reset! Step: {steps}, Total Reward: {total_reward:.2f}")
                elif event.key in action_map:
                    action = action_map[event.key]
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                    
                    if terminated or truncated:
                        print(f"Episode finished! Final reward: {total_reward:.2f}")
                        if info.get('collected_rewards', 0) == env.num_rewards:
                            print("🎉 Mission accomplished! All rewards collected!")
                        state, info = env.reset()
                        total_reward = 0
                        steps = 0
        
        env.render()
        time.sleep(0.1)
    
    env.close()
    pygame.quit()


def create_environment(args) -> Any:
    """Factory function to create environment based on type"""
    if args.env_type == "gridworld":
        return GridWorldEnv(
            grid_size=args.grid_size,
            num_rewards=args.num_rewards,
            num_enemies=args.num_enemies,
            num_obstacles=args.num_obstacles,
            render_mode="human" if hasattr(args, 'render') and args.render else None,
            fixed_layout=args.fixed_layout,
            use_reward_shaping=getattr(args, 'use_reward_shaping', True)
        )
    elif args.env_type == "movingenemyworld":
        return MovingEnemyGridWorldEnv(
            grid_size=args.grid_size,
            num_rewards=args.num_rewards,
            num_enemies=args.num_enemies,
            num_obstacles=args.num_obstacles,
            render_mode="human" if hasattr(args, 'render') and args.render else None,
            fixed_layout=args.fixed_layout,
            enemy_move_probability=getattr(args, 'enemy_move_probability', 0.8),
            use_reward_shaping=getattr(args, 'use_reward_shaping', True)
        )
    else:
        raise ValueError(f"Unknown environment type: {args.env_type}")


def get_state_size(env_type: str, args) -> int:
    """Get state size for DQN based on environment type"""
    if env_type in ["gridworld", "movingenemyworld"]:
        # GridWorld environment observation structure
        # agent_pos (2) + reward_vectors (num_rewards * 2) + enemy_vectors (num_enemies * 2) + obstacle_distances (8)
        return 2 + args.num_rewards * 2 + args.num_enemies * 2 + 8
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def create_agent(env_type: str, args) -> DQNAgent:
    """Factory function to create DQN agent based on environment type"""
    state_size = get_state_size(env_type, args)
    
    # Get training parameters with defaults for testing mode
    learning_rate = getattr(args, 'learning_rate', 0.001)
    gamma = getattr(args, 'gamma', 0.99)
    epsilon = getattr(args, 'epsilon', 1.0)
    epsilon_min = getattr(args, 'epsilon_min', 0.01)
    epsilon_decay = getattr(args, 'epsilon_decay', 0.995)
    buffer_size = getattr(args, 'buffer_size', 10000)
    batch_size = getattr(args, 'batch_size', 64)
    target_update = getattr(args, 'target_update', 100)
    
    if env_type in ["gridworld", "movingenemyworld"]:
        # GridWorld-specific DQN agent (works for both static and moving enemy versions)
        return DQNAgent(
            state_size=state_size,
            lr=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update=target_update
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def train_agent(args):
    """Train a DQN agent on the specified environment"""
    print("🚀 Training DQN Agent")
    
    # Create environment
    env = create_environment(args)
    
    # Create agent
    agent = create_agent(args.env_type, args)
    
    # Get environment config for training
    env_config = get_env_config(args)
    
    # Train the agent
    episode_rewards, episode_lengths = agent.train(
        env_config=env_config,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_dir="models",
        save_interval=args.save_interval,
        print_interval=args.print_interval
    )
    
    # Save final model
    final_model_path = f"models/{args.env_type}_dqnagent_final.pth"
    agent.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Optional: Plot training progress
    if args.plot:
        agent.plot_training_progress()
    
    # Optional: Evaluate the trained agent
    if args.evaluate:
        print("\n📊 Evaluating trained agent...")
        avg_reward, success_rate = agent.evaluate(
            env_config=env_config,
            episodes=args.eval_episodes,
            render=args.render_eval
        )
        print(f"Evaluation Results: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1%}")


def test_agent(args):
    """Test a trained DQN agent"""
    print("🧪 Testing Trained Agent")
    
    # Create environment
    env = create_environment(args)
    
    # Create agent
    agent = create_agent(args.env_type, args)
    
    # Set default model path if not provided
    if args.model is None:
        args.model = f"models/{args.env_type}_dqnagent_final.pth"
    
    # Load trained model
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"📂 Model loaded: {args.model}")
    else:
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Get environment config for evaluation
    env_config = get_env_config(args)
    
    # Evaluate the agent
    avg_reward, success_rate = agent.evaluate(
        env_config=env_config,
        episodes=args.eval_episodes,
        render=args.render,
        max_steps=args.max_steps
    )
    
    print(f"\n📊 Test Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1%}")


def get_env_config(args) -> Dict[str, Any]:
    """Get environment configuration based on type"""
    if args.env_type == "gridworld":
        return {
            "env_type": "gridworld",
            "grid_size": args.grid_size,
            "num_rewards": args.num_rewards,
            "num_enemies": args.num_enemies,
            "num_obstacles": args.num_obstacles,
            "render_mode": None,
            "fixed_layout": args.fixed_layout,
            "use_reward_shaping": getattr(args, 'use_reward_shaping', True)
        }
    elif args.env_type == "movingenemyworld":
        return {
            "env_type": "movingenemyworld",
            "grid_size": args.grid_size,
            "num_rewards": args.num_rewards,
            "num_enemies": args.num_enemies,
            "num_obstacles": args.num_obstacles,
            "render_mode": None,
            "fixed_layout": args.fixed_layout,
            "enemy_move_probability": getattr(args, 'enemy_move_probability', 0.8),
            "use_reward_shaping": getattr(args, 'use_reward_shaping', True)
        }
    else:
        raise ValueError(f"Unknown environment type: {args.env_type}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Multi-Environment Deep RL Framework")
    
    # Add environment type selection
    parser.add_argument('--env-type', choices=['gridworld', 'movingenemyworld'], default='gridworld',
                       help='Environment type (default: gridworld)')
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Play mode
    play_parser = subparsers.add_parser('play', help='Human control mode')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train DQN agent')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--save-interval', type=int, default=200, help='Save model every N episodes')
    train_parser.add_argument('--print-interval', type=int, default=50, help='Print progress every N episodes')
    train_parser.add_argument('--plot', action='store_true', help='Plot training progress')
    train_parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    train_parser.add_argument('--render-eval', action='store_true', help='Render during evaluation')
    
    # Test mode
    test_parser = subparsers.add_parser('test', help='Test trained agent')
    test_parser.add_argument('--model', type=str, default=None, 
                            help='Path to model file (.pth) - defaults to models/{env_type}_dqnagent_final.pth')
    test_parser.add_argument('--render', action='store_true', help='Render during testing')
    
    # Add common arguments to all subparsers
    for subparser in [play_parser, train_parser, test_parser]:
        subparser.add_argument('--env-type', choices=['gridworld', 'movingenemyworld'], default='gridworld',
                              help='Environment type (default: gridworld)')
        subparser.add_argument('--eval-episodes', type=int, default=10, help='Episodes for evaluation')
        
        # GridWorld-specific arguments
        subparser.add_argument('--grid-size', type=int, default=8, help='Grid size (NxN)')
        subparser.add_argument('--num-rewards', type=int, default=2, help='Number of rewards')
        subparser.add_argument('--num-enemies', type=int, default=4, help='Number of enemies')
        subparser.add_argument('--num-obstacles', type=int, default=8, help='Number of obstacles')
        subparser.add_argument('--max-steps', type=int, default=192, help='Max steps per episode')
        subparser.add_argument('--fixed-layout', action='store_true', default=True, 
                              help='Use fixed layout (default: True)')
        subparser.add_argument('--random-layout', dest='fixed_layout', action='store_false',
                              help='Use random layout instead of fixed')
        
        # MovingEnemyWorld-specific arguments
        subparser.add_argument('--enemy-move-probability', type=float, default=0.8,
                              help='Probability that enemies move each step (default: 0.8)')
        
        # Reward shaping arguments
        subparser.add_argument('--use-reward-shaping', action='store_true', default=True,
                              help='Use potential-based reward shaping (default: True)')
        subparser.add_argument('--no-reward-shaping', dest='use_reward_shaping', action='store_false',
                              help='Disable reward shaping (use simple distance rewards)')
    
    # Training-specific arguments
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    train_parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    train_parser.add_argument('--target-update', type=int, default=100, help='Target network update frequency')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    print("🧠 Multi-Environment Deep RL Framework")
    print("=" * 45)
    print(f"Environment: {args.env_type.upper()}")
    if args.env_type in ["gridworld", "movingenemyworld"]:
        print(f"Config: {args.grid_size}x{args.grid_size} {'Fixed' if args.fixed_layout else 'Random'} Layout")
        print(f"Objects: {args.num_rewards} Rewards | {args.num_enemies} Enemies | {args.num_obstacles} Obstacles")
        if args.env_type == "movingenemyworld":
            print(f"Enemy Movement: {args.enemy_move_probability*100:.0f}% probability each step")
        print(f"Reward Shaping: {'✅ Enabled' if getattr(args, 'use_reward_shaping', True) else '❌ Disabled'}")
    print("=" * 45)
    
    os.makedirs("models", exist_ok=True)
    
    if args.mode == 'play':
        play_human(args)
    elif args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args)


if __name__ == "__main__":
    main() 