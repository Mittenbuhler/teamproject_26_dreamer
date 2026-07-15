"""ausführen mit command: .venv/bin/python -m sprint5.main"""

import gymnasium as gym
import torch
import torch.optim as optim
from dataclasses import dataclass

from .models import RSSM, Actor, Critic, DEVICE
from .data import ReplayBuffer, collect_episodes, render_frame, select_action
from .train import train_world_model, train_actor_critic
from pathlib import Path

def evaluate_policy(env, world_model, actor, episodes=5, max_steps=200):
    actor.eval()
    world_model.eval()
    returns = []
    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=ep_idx)
        flatz, h = world_model.initial(batch_size=1, device=DEVICE)
        frame = render_frame(env)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            a, a_idx = select_action(actor, flatz, h, epsilon=0.0)
            obs, r, terminated, truncated, _ = env.step(a_idx)
            done = terminated or truncated
            total_reward += r

            frame = render_frame(env)
            next_vis = torch.tensor(frame, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flatz, h = world_model.step_online(flatz, h, a, next_vis)

            steps += 1
        returns.append(total_reward)
    return sum(returns) / len(returns)


LR_WORLD = 1e-3
LR_AC = 1e-4


@dataclass
class TrainConfig:
    seq_len: int = 16
    warmup_steps: int = 5
    imagination_horizon: int = 8
    batch_size: int = 8
    epochs_per_phase: int = 1
    initial_random_episodes: int = 10
    initial_world_model_epochs: int = 3
    initial_ac_pretrain_iters: int = 3
    initial_ac_pretrain_episodes: int = 4
    actor_entropy_coeff: float = 0.05
    ac_sample_episodes: int = 4
    clip_grad_norm: float = 0.5
    collect_episodes_per_iter: int = 4
    max_steps: int = 200
    iterations: int = 20
    replay_capacity: int = 300
    train_interval_episodes: int = 4
    explore_epsilon: float = 0.05
    eval_episodes: int = 20


def build_actor_critic(world_model):
    feat_size = world_model.stoch_size + world_model.h_size
    actor = Actor(feat_size).to(DEVICE)
    critic = Critic(feat_size).to(DEVICE)
    return actor, critic


def iterative_train(cfg=TrainConfig()):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(cfg.replay_capacity)

    world_model = RSSM().to(DEVICE)
    actor, critic = build_actor_critic(world_model)

    wm_opt = optim.Adam(world_model.parameters(), lr=LR_WORLD)
    actor_opt = optim.Adam(actor.parameters(), lr=LR_AC, weight_decay=1e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_AC, weight_decay=1e-5)

    for ep in collect_episodes(env, world_model, actor, n_episodes=cfg.initial_random_episodes, max_steps=cfg.max_steps, seed=42, epsilon=1.0):
        replay_buffer.add_episode(ep)

    if cfg.initial_world_model_epochs > 0:
        init_wm_metrics = train_world_model(
            world_model,
            wm_opt,
            replay_buffer,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            epochs_per_phase=cfg.initial_world_model_epochs,
        )
        print(f"init_wm buffer={len(replay_buffer):03d} init_wm={init_wm_metrics}")

    for pre_it in range(cfg.initial_ac_pretrain_iters):
        new_eps = collect_episodes(
            env,
            world_model,
            actor,
            n_episodes=cfg.initial_ac_pretrain_episodes,
            max_steps=cfg.max_steps,
            epsilon=cfg.explore_epsilon,
        )
        for ep in new_eps:
            replay_buffer.add_episode(ep)

        pre_ac_metrics = train_actor_critic(
            world_model,
            actor,
            critic,
            actor_opt,
            critic_opt,
            replay_buffer,
            imagination_horizon=cfg.imagination_horizon,
            warmup_steps=cfg.warmup_steps,
            sample_episodes=cfg.ac_sample_episodes,
            entropy_coeff=cfg.actor_entropy_coeff,
            clip_norm=cfg.clip_grad_norm,
        )
        pre_eval_return = evaluate_policy(env, world_model, actor, episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
        print(f"pretrain={pre_it+1}/{cfg.initial_ac_pretrain_iters} buffer={len(replay_buffer):03d} ac={pre_ac_metrics} eval_return={pre_eval_return:.2f}")

    for it in range(cfg.iterations):
        new_eps = collect_episodes(
            env,
            world_model,
            actor,
            n_episodes=cfg.collect_episodes_per_iter,
            max_steps=cfg.max_steps,
            epsilon=cfg.explore_epsilon,
        )
        for ep in new_eps:
            replay_buffer.add_episode(ep)

        wm_metrics = train_world_model(
            world_model,
            wm_opt,
            replay_buffer,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            epochs_per_phase=cfg.epochs_per_phase,
        )
        ac_metrics = train_actor_critic(
            world_model,
            actor,
            critic,
            actor_opt,
            critic_opt,
            replay_buffer,
            imagination_horizon=cfg.imagination_horizon,
            warmup_steps=cfg.warmup_steps,
            sample_episodes=cfg.ac_sample_episodes,
            entropy_coeff=cfg.actor_entropy_coeff,
            clip_norm=cfg.clip_grad_norm,
        )

        eval_return = evaluate_policy(env, world_model, actor, episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
        print(f"iter={it:03d} buffer={len(replay_buffer):03d} wm={wm_metrics} ac={ac_metrics} eval_return={eval_return:.2f}")

    SCRIPT_DIR = Path(__file__).resolve().parent
    torch.save(world_model.state_dict(), SCRIPT_DIR / "world_model.pth")
    torch.save(actor.state_dict(), SCRIPT_DIR / "actor.pth")
    torch.save(critic.state_dict(), SCRIPT_DIR / "critic.pth")

    env.close()
    return world_model, actor, critic


if __name__ == "__main__":
    iterative_train()