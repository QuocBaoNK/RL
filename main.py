#!/usr/bin/env python3
"""
Simplified Grid World RL Main Script
Supports training, testing, and playing with DQN and Q-Table agents in Grid World environment
Default: 5x5 fixed layout with 1 reward and 3 enemies
"""

import argparse
import os
import sys
import pygame
import time
from typing import Optional

from environment import GridWorldEnv
from agents import DQNAgent, QTableAgent


def play_human(args):
    """Human control mode"""
    print("🎮 Human Control Mode")
    print("Controls: W/S/A/D or Arrow Keys, ESC to quit")
    print(f"Environment: {args.grid_size}x{args.grid_size} {'Fixed' if args.fixed_layout else 'Random'} Layout")
    
    env = GridWorldEnv(
        grid_size=args.grid_size,
        num_rewards=args.num_rewards,
        num_enemies=args.num_enemies,
        render_mode="human",
        fixed_layout=args.fixed_layout
    )
    
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
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key in action_map:
                    action = action_map[event.key]
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    print(f"Step {steps}: Reward={reward:.2f}, Total={total_reward:.2f}")
                    
                    if terminated or truncated:
                        if info.get('remaining_rewards', 1) == 0:
                            print("🎉 SUCCESS! All rewards collected!")
                        else:
                            print("💀 Game Over!")
                        print(f"Final Score: {total_reward:.2f} in {steps} steps")
                        running = False
                elif event.key == pygame.K_r:
                    state, info = env.reset()
                    env.print_layout()
                    total_reward = 0
                    steps = 0
                    print("Game reset!")
        
        env.render()
    
    env.close()
    pygame.quit()


def train_agent(args):
    """Train a new agent (DQN or Q-Table)"""
    print(f"🤖 Training {args.agent_type.upper()} Agent for {args.episodes} episodes")
    print(f"Environment: {args.grid_size}x{args.grid_size} {'Fixed' if args.fixed_layout else 'Random'} Layout")
    
    env_config = {
        "grid_size": args.grid_size,
        "num_rewards": args.num_rewards,
        "num_enemies": args.num_enemies,
        "render_mode": None,
        "fixed_layout": args.fixed_layout
    }
    
    temp_env = GridWorldEnv(**env_config)
    temp_env.reset()
    temp_env.print_layout()
    temp_env.close()
    
    if args.agent_type == "dqn":
        state_size = 2 + 2 * args.grid_size * args.grid_size
        agent = DQNAgent(
            state_size=state_size,
            lr=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update=args.target_update
        )
    else:
        agent = QTableAgent(
            lr=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay
        )
    
    episode_rewards, episode_lengths = agent.train(
        env_config=env_config,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_interval=args.save_interval,
        print_interval=args.print_interval
    )
    
    print("✅ Training completed!")
    
    if args.plot:
        plot_name = f"{args.agent_type}_training_progress.png"
        agent.plot_training_progress(plot_name)
    
    if args.agent_type == "qtable" and hasattr(agent, 'print_q_table_sample'):
        print("\n" + "="*50)
        agent.print_q_table_sample(5)
    
    if args.evaluate:
        print("\n🧪 Evaluating trained agent...")
        avg_reward, success_rate = agent.evaluate(
            env_config=env_config,
            episodes=args.eval_episodes, 
            render=args.render_eval,
            max_steps=args.max_steps
        )


def test_agent(args):
    """Test a trained agent"""
    print(f"🧪 Testing trained agent: {args.model}")
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    env_config = {
        "grid_size": args.grid_size,
        "num_rewards": args.num_rewards,
        "num_enemies": args.num_enemies,
        "render_mode": "human" if args.render else None,
        "fixed_layout": args.fixed_layout
    }
    
    if args.model.endswith('.pkl'):
        print("Loading Q-Table agent...")
        agent = QTableAgent()
    else:
        print("Loading DQN agent...")
        state_size = 2 + 2 * args.grid_size * args.grid_size
        agent = DQNAgent(state_size=state_size)
    
    agent.load(args.model)
    
    avg_reward, success_rate = agent.evaluate(
        env_config=env_config,
        episodes=args.eval_episodes,
        render=args.render,
        max_steps=args.max_steps
    )


def compare_agents(args):
    """Compare DQN vs Q-Table agents"""
    print("🔬 Comparing DQN vs Q-Table Agents")
    print(f"Environment: {args.grid_size}x{args.grid_size} {'Fixed' if args.fixed_layout else 'Random'} Layout")
    print("=" * 50)
    
    env_config = {
        "grid_size": args.grid_size,
        "num_rewards": args.num_rewards,
        "num_enemies": args.num_enemies,
        "render_mode": None,
        "fixed_layout": args.fixed_layout
    }
    
    results = {}
    
    dqn_model = "models/dqnagent_final.pth"
    if os.path.exists(dqn_model):
        print("Testing DQN Agent...")
        state_size = 2 + 2 * args.grid_size * args.grid_size
        dqn_agent = DQNAgent(state_size=state_size)
        dqn_agent.load(dqn_model)
        dqn_avg, dqn_success = dqn_agent.evaluate(
            env_config=env_config,
            episodes=args.eval_episodes, 
            render=False, 
            max_steps=args.max_steps
        )
        results["DQN"] = {"avg_reward": dqn_avg, "success_rate": dqn_success}
    
    qtable_model = "models/qtableagent_final.pkl"
    if os.path.exists(qtable_model):
        print("\nTesting Q-Table Agent...")
        qtable_agent = QTableAgent()
        qtable_agent.load(qtable_model)
        qtable_avg, qtable_success = qtable_agent.evaluate(
            env_config=env_config,
            episodes=args.eval_episodes, 
            render=False, 
            max_steps=args.max_steps
        )
        results["Q-Table"] = {"avg_reward": qtable_avg, "success_rate": qtable_success}
    
    if results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        for agent_type, result in results.items():
            print(f"\n{agent_type} Agent:")
            print(f"  Average Reward: {result['avg_reward']:.2f}")
            print(f"  Success Rate: {result['success_rate']:.2%}")
        
        if len(results) > 1:
            best_agent = max(results.items(), key=lambda x: x[1]['success_rate'])
            print(f"\n🏆 Best Agent: {best_agent[0]}")
            print(f"   Success Rate: {best_agent[1]['success_rate']:.2%}")
    else:
        print("❌ No trained models found! Train agents first.")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Grid World RL with DQN and Q-Table")
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    play_parser = subparsers.add_parser('play', help='Human control mode')
    
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--agent-type', choices=['dqn', 'qtable'], default='qtable', 
                             help='Type of agent to train (default: qtable for 5x5 grid)')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--save-interval', type=int, default=200, help='Save model every N episodes')
    train_parser.add_argument('--print-interval', type=int, default=50, help='Print progress every N episodes')
    train_parser.add_argument('--plot', action='store_true', help='Plot training progress')
    train_parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    train_parser.add_argument('--render-eval', action='store_true', help='Render during evaluation')
    
    test_parser = subparsers.add_parser('test', help='Test trained agent')
    test_parser.add_argument('--model', type=str, default='models/qtableagent_final.pkl', 
                            help='Path to model file (.pth for DQN, .pkl for Q-Table)')
    test_parser.add_argument('--render', action='store_true', help='Render during testing')
    
    compare_parser = subparsers.add_parser('compare', help='Compare DQN vs Q-Table agents')
    
    for subparser in [play_parser, train_parser, test_parser, compare_parser]:
        subparser.add_argument('--grid-size', type=int, default=5, help='Grid size (NxN)')
        subparser.add_argument('--num-rewards', type=int, default=1, help='Number of rewards')
        subparser.add_argument('--num-enemies', type=int, default=3, help='Number of enemies')
        subparser.add_argument('--max-steps', type=int, default=75, help='Max steps per episode (5x5x3)')
        subparser.add_argument('--eval-episodes', type=int, default=10, help='Episodes for evaluation')
        subparser.add_argument('--fixed-layout', action='store_true', default=True, 
                              help='Use fixed layout (default: True)')
        subparser.add_argument('--random-layout', dest='fixed_layout', action='store_false',
                              help='Use random layout instead of fixed')
    
    train_parser.add_argument('--learning-rate', type=float, default=0.1, 
                             help='Learning rate (0.001 for DQN, 0.1 for Q-Table)')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    
    train_parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size (DQN only)')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (DQN only)')
    train_parser.add_argument('--target-update', type=int, default=100, 
                             help='Target network update frequency (DQN only)')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    print("🎮 Grid World RL with DQN and Q-Table")
    print("=" * 45)
    print(f"Default: 5x5 Fixed Layout | 1 Reward | 3 Enemies")
    print("=" * 45)
    
    os.makedirs("models", exist_ok=True)
    
    if hasattr(args, 'agent_type') and args.agent_type == 'dqn' and args.learning_rate == 0.1:
        args.learning_rate = 0.001
        print(f"ℹ️  Adjusted learning rate to {args.learning_rate} for DQN agent")
    
    if args.mode == 'play':
        play_human(args)
    elif args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args)
    elif args.mode == 'compare':
        compare_agents(args)


if __name__ == "__main__":
    main() 