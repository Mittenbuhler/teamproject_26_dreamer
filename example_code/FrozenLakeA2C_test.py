"""
Tests für FrozenLakeA2C.py

In das richtige Verzeichnis wechseln: teamproject_26_dreamer

Dann ausführen mit:
python3 -m pytest example_code/FrozenLakeA2C_test.py -v
"""

import os
import pytest
import numpy as np
import torch
import gymnasium as gym

# WICHTIG:
# Verhindert, dass beim Import direkt GIFs angezeigt oder gespeichert werden
from unittest.mock import patch
import importlib

with patch("IPython.display.display"), \
     patch("imageio.mimsave"):

    frozenlake_module = importlib.import_module(
        "example_code.FrozenLakeA2C"
    )

ActorCritic = frozenlake_module.ActorCritic
A2CAgentFrozenLake = frozenlake_module.A2CAgentFrozenLake
record_episode = frozenlake_module.record_episode


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    """
    Erstellt eine frische FrozenLake-Umgebung.
    """
    e = gym.make("FrozenLake-v1", is_slippery=False)
    yield e
    e.close()


@pytest.fixture
def model(env):
    """
    Kleines Netzwerk für schnelle Tests.
    """
    return ActorCritic(
        hidden_size=32,
        num_outputs=env.action_space.n
    )


@pytest.fixture
def agent(env):
    """
    Agent mit minimalem Training.
    """
    return A2CAgentFrozenLake(
        env,
        num_episodes=5,
        max_steps=20,
        hidden_size=32
    )


# ══════════════════════════════════════════════════════════════════════════════
# WHITEBOX-TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestActorCriticNetwork:
    """
    Prüft interne Eigenschaften des Netzwerks.
    """

    def test_output_shapes_single_state(self, model):
        """
        Für einen einzelnen Zustand:
        action_probs -> (4,)
        value -> (1,)
        """
        state = torch.FloatTensor(np.eye(16)[0])

        action_probs, value = model(state)

        assert action_probs.shape == (4,)
        assert value.shape == (1,)

    def test_output_shapes_batched(self, model):
        """
        Batch aus 8 Zuständen:
        action_probs -> (8,4)
        value -> (8,1)
        """
        batch = torch.FloatTensor(
            np.eye(16)[np.random.randint(0, 16, size=8)]
        )

        action_probs, value = model(batch)

        assert action_probs.shape == (8, 4)
        assert value.shape == (8, 1)

    def test_action_probs_sum_to_one(self, model):
        """
        Softmax-Ausgabe muss sich zu 1 summieren.
        """
        state = torch.FloatTensor(np.eye(16)[3])

        action_probs, _ = model(state)

        assert torch.isclose(
            action_probs.sum(),
            torch.tensor(1.0),
            atol=1e-6
        )

    def test_action_probs_non_negative(self, model):
        """
        Keine negativen Wahrscheinlichkeiten erlaubt.
        """
        state = torch.FloatTensor(np.eye(16)[5])

        action_probs, _ = model(state)

        assert (action_probs >= 0).all()

    def test_value_is_scalar(self, model):
        """
        Critic muss genau einen Wert ausgeben.
        """
        state = torch.FloatTensor(np.eye(16)[2])

        _, value = model(state)

        assert value.numel() == 1

    def test_different_states_produce_different_outputs(self, model):
        """
        Unterschiedliche Zustände sollen unterschiedliche
        Policy-Ausgaben erzeugen.
        """
        s1 = torch.FloatTensor(np.eye(16)[1])
        s2 = torch.FloatTensor(np.eye(16)[14])

        p1, _ = model(s1)
        p2, _ = model(s2)

        assert not torch.allclose(p1, p2)

    def test_gradients_flow_through_network(self, model):
        """
        Nach backward() müssen Gradienten vorhanden sein.
        """
        state = torch.FloatTensor(np.eye(16)[0])

        action_probs, value = model(state)

        loss = -torch.log(action_probs[0]) + value.squeeze()

        loss.backward()

        assert model.hidden_layer.weight.grad is not None
        assert model.actor_layer.weight.grad is not None
        assert model.critic_layer.weight.grad is not None


class TestComputeReturns:
    """
    Prüft die Return-Berechnung.
    """

    def test_single_reward(self, agent):
        """
        Ein Reward -> gleicher Return.
        """
        returns = agent.compute_returns([1.0])

        assert torch.isclose(
            returns[0],
            torch.tensor(1.0)
        )

    def test_two_rewards_manual(self, agent):
        """
        Manuelle Return-Berechnung.
        """
        returns = agent.compute_returns([1.0, 1.0])

        expected_0 = 1.0 + agent.gamma * 1.0
        expected_1 = 1.0

        assert torch.isclose(
            returns[0],
            torch.tensor(expected_0),
            atol=1e-5
        )

        assert torch.isclose(
            returns[1],
            torch.tensor(expected_1),
            atol=1e-5
        )

    def test_returns_length_matches_rewards(self, agent):
        """
        Returns und Rewards müssen gleiche Länge haben.
        """
        rewards = [0, 0, 1, 0]

        returns = agent.compute_returns(rewards)

        assert len(returns) == len(rewards)

    def test_zero_rewards(self, agent):
        """
        Nur Null-Rewards -> nur Null-Returns.
        """
        returns = agent.compute_returns([0, 0, 0])

        assert torch.all(returns == 0)

    def test_returns_on_correct_device(self, agent):
        """
        Returns müssen auf dem richtigen Device liegen.
        """
        returns = agent.compute_returns([1, 0, 1])

        assert returns.device.type == agent.device.type


class TestTrainingLoop:
    """
    Tests für den Trainingsprozess.
    """

    def test_parameters_change_after_training(self, env):
        """
        Nach dem Training müssen sich Gewichte ändern.
        """
        agent = A2CAgentFrozenLake(
            env,
            num_episodes=10,
            max_steps=20,
            hidden_size=32
        )

        params_before = [
            p.clone()
            for p in agent.policy_net.parameters()
        ]

        agent.train()

        params_after = list(agent.policy_net.parameters())

        changed = any(
            not torch.equal(b, a)
            for b, a in zip(params_before, params_after)
        )

        assert changed

    def test_train_returns_correct_length(self, env):
        """
        train() muss num_episodes Rewards zurückgeben.
        """
        num_episodes = 7

        agent = A2CAgentFrozenLake(
            env,
            num_episodes=num_episodes,
            max_steps=20,
            hidden_size=32
        )

        rewards = agent.train()

        assert len(rewards) == num_episodes


# ══════════════════════════════════════════════════════════════════════════════
# BLACKBOX-TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentBehavior:
    """
    Testet sichtbares Verhalten des Agenten.
    """

    def test_action_is_valid(self, agent, env):
        """
        Aktionen müssen im Bereich [0,1,2,3] liegen.
        """
        state, _ = env.reset()

        for _ in range(20):

            state_tensor = torch.FloatTensor(
                np.eye(16)[state]
            )

            action_probs, _ = agent.policy_net(state_tensor)

            action = torch.multinomial(
                action_probs,
                1
            ).item()

            assert action in [0, 1, 2, 3]

            state, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                state, _ = env.reset()

    def test_train_returns_numpy_array(self, env):
        """
        train() soll np.ndarray liefern.
        """
        agent = A2CAgentFrozenLake(
            env,
            num_episodes=5,
            max_steps=20,
            hidden_size=32
        )

        rewards = agent.train()

        assert isinstance(rewards, np.ndarray)

    def test_rewards_are_binary(self, env):
        """
        FrozenLake liefert nur Rewards 0 oder 1.
        """
        agent = A2CAgentFrozenLake(
            env,
            num_episodes=10,
            max_steps=20,
            hidden_size=32
        )

        rewards = agent.train()

        assert np.all(np.isin(rewards, [0, 1]))

    def test_action_probs_sum_to_one_on_real_state(self, agent, env):
        """
        Auch echte Zustände müssen gültige
        Wahrscheinlichkeitsverteilungen liefern.
        """
        state, _ = env.reset()

        state_tensor = torch.FloatTensor(
            np.eye(16)[state]
        )

        action_probs, _ = agent.policy_net(state_tensor)

        assert torch.isclose(
            action_probs.sum(),
            torch.tensor(1.0),
            atol=1e-5
        )


class TestGifRecording:
    """
    Tests für GIF-Erstellung.
    """

    def test_record_episode_returns_valid_reward(self, agent):

        env = gym.make(
            "FrozenLake-v1",
            is_slippery=False,
            render_mode="rgb_array"
        )

        with patch("imageio.mimsave"):

            _, reward = record_episode(
                agent,
                env,
                use_legacy_render=False,
                output_path="temp.gif"
            )

        assert reward in [0, 1]

        env.close()