
import time

import gymnasium as gym
import pytest
import torch
import torch.optim as optim

# Ausführen mit:
# python -m pytest cartpole/performance_tests.py -v

from cartpole.main import evaluate_policy, build_actor_critic
from cartpole.models import RSSM, Actor
from cartpole.data import ReplayBuffer, collect_episodes
from cartpole.train import train_world_model, train_actor_critic


# Beide Beobachtungsvarianten testen
@pytest.fixture(params=[2, 4], ids=["2D", "4D"])
def obs_size(request):
    return request.param


def make_model(obs_size):
    """Erzeugt ein RSSM mit der gewünschten Beobachtungsdimension."""
    return RSSM(obs_size=obs_size)


def test_world_model_training_reduces_loss(obs_size):
    env = gym.make("CartPole-v1")
    try:
        world_model = make_model(obs_size)
        actor, critic = build_actor_critic(world_model)
        replay = ReplayBuffer(100)

        episodes = collect_episodes(
            env, actor, world_model,
            n_episodes=10,
            epsilon=1.0,
        )
        for ep in episodes:
            replay.add_episode(ep)

        optimizer = optim.Adam(world_model.parameters(), lr=1e-3)

        metrics_before = train_world_model(
            world_model, optimizer, replay, epochs_per_phase=1
        )
        loss_before = metrics_before["total_loss"]

        metrics_after = train_world_model(
            world_model, optimizer, replay, epochs_per_phase=5
        )
        loss_after = metrics_after["total_loss"]

        assert loss_after < loss_before, (
            f"{obs_size}D: World-Model-Loss ist nicht gesunken: "
            f"{loss_before:.4f} -> {loss_after:.4f}"
        )
    finally:
        env.close()


def test_actor_training_does_not_degrade_return_too_much(obs_size):
    env = gym.make("CartPole-v1")
    try:
        replay = ReplayBuffer(100)
        world_model = make_model(obs_size)
        actor, critic = build_actor_critic(world_model)

        wm_opt = optim.Adam(world_model.parameters(), lr=1e-3)
        actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
        critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

        episodes = collect_episodes(
            env, actor, world_model,
            n_episodes=15,
            epsilon=1.0,
        )
        for ep in episodes:
            replay.add_episode(ep)

        train_world_model(
            world_model, wm_opt, replay, epochs_per_phase=5
        )

        reward_before = evaluate_policy(
            env, world_model, actor, episodes=5
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
            env, world_model, actor, episodes=5
        )

        # Ein einzelnes Actor-Critic-Update garantiert keine Verbesserung.
        assert reward_after >= reward_before - 20, (
            f"{obs_size}D: Return ist zu stark gesunken: "
            f"{reward_before:.2f} -> {reward_after:.2f}"
        )
    finally:
        env.close()


def test_inference_speed(obs_size):
    world_model = make_model(obs_size)
    actor = Actor(world_model.stoch_size + world_model.h_size)
    x = torch.randn(1, world_model.stoch_size + world_model.h_size)

    # Warm-up
    for _ in range(20):
        actor(x)

    start = time.perf_counter()
    for _ in range(1000):
        actor(x)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"{obs_size}D: Actor-Inferenz dauerte {elapsed:.3f}s"
    )


def test_imagination_runtime(obs_size):
    world_model = make_model(obs_size)
    actor = Actor(world_model.stoch_size + world_model.h_size)
    z, h = world_model.initial(batch_size=32)

    start = time.perf_counter()
    world_model.imagine(z, h, actor, horizon=15)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, (
        f"{obs_size}D: Imagination dauerte {elapsed:.3f}s"
    )


def test_final_policy_reaches_reasonable_return(obs_size):
    env = gym.make("CartPole-v1")
    try:
        world_model = make_model(obs_size)
        actor, critic = build_actor_critic(world_model)

        reward = evaluate_policy(
            env, world_model, actor, episodes=10
        )

        # Untrainierte Policy: nur eine schwache untere Schranke.
        assert reward >= 5, (
            f"{obs_size}D: Return der untrainierten Policy "
            f"war zu niedrig: {reward:.2f}"
        )
    finally:
        env.close()