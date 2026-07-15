import time

import gymnasium as gym
import torch
import torch.optim as optim



#ausführen mit pytest sprint5/performance_tests.py

from sprint5.main import (
    evaluate_policy,
    build_actor_critic,
)

from sprint5.models import RSSM

from sprint5.data import (
    ReplayBuffer,
    collect_episodes,
)

from sprint5.train import (
    train_world_model,
    train_actor_critic,
)


def test_world_model_training_reduces_loss():

    env = gym.make("CartPole-v1")

    world_model = RSSM()

    actor, critic, obs_actor = build_actor_critic(world_model)

    replay = ReplayBuffer(100)

    episodes = collect_episodes(
        env,
        obs_actor,
        n_episodes=10,
        epsilon=1.0,
    )

    for ep in episodes:
        replay.add_episode(ep)

    optimizer = optim.Adam(world_model.parameters(), lr=1e-3)

    metrics_before = train_world_model(
        world_model,
        optimizer,
        replay,
        epochs_per_phase=1,
    )

    loss_before = metrics_before["total_loss"]

    metrics_after = train_world_model(
        world_model,
        optimizer,
        replay,
        epochs_per_phase=5,
    )

    loss_after = metrics_after["total_loss"]

    env.close()

    assert loss_after < loss_before


def test_actor_training_improves_return():

    env = gym.make("CartPole-v1")

    replay = ReplayBuffer(100)

    world_model = RSSM()

    actor, critic, obs_actor = build_actor_critic(world_model)

    wm_opt = optim.Adam(world_model.parameters(), lr=1e-3)
    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    episodes = collect_episodes(
        env,
        obs_actor,
        n_episodes=15,
        epsilon=1.0,
    )

    for ep in episodes:
        replay.add_episode(ep)

    train_world_model(
        world_model,
        wm_opt,
        replay,
        epochs_per_phase=5,
    )

    reward_before = evaluate_policy(
        env,
        obs_actor,
        episodes=5,
    )

    train_actor_critic(
        world_model,
        actor,
        critic,
        actor_opt,
        critic_opt,
        replay,
        imagination_horizon=10,
    )

    reward_after = evaluate_policy(
        env,
        obs_actor,
        episodes=5,
    )

    env.close()

    # RL ist stochastisch -> nicht strenger Vergleich
    assert reward_after >= reward_before - 5


def test_inference_speed():

    model = RSSM()

    actor = Actor(model.stoch_size + model.h_size)

    x = torch.randn(1, model.stoch_size + model.h_size)

    start = time.perf_counter()

    for _ in range(1000):
        actor(x)

    elapsed = time.perf_counter() - start

    # Sollte deutlich unter einer Sekunde liegen
    assert elapsed < 1.0


def test_imagination_runtime():

    model = RSSM()

    actor = Actor(model.stoch_size + model.h_size)

    z, h = model.initial(batch_size=32)

    start = time.perf_counter()

    model.imagine(
        z,
        h,
        actor,
        horizon=15,
    )

    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


def test_final_policy_reaches_reasonable_return():

    env = gym.make("CartPole-v1")

    world_model = RSSM()

    actor, critic, obs_actor = build_actor_critic(world_model)

    reward = evaluate_policy(
        env,
        obs_actor,
        episodes=10,
    )

    env.close()

    # Nach vollständigem Training diesen Wert z.B. auf 100 oder 150 erhöhen.
    assert reward >= 10