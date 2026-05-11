"""
Tests für CartPoleA2C.py
Ausführen mit: pytest example_code/CartPoleA2C_test.py -v
"""

import os
import pytest
import numpy as np
import torch
import gymnasium as gym

from example_code.CartPoleA2C import ActorCritic, A2CAgentCartPole, record_episode


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES  – wiederverwendbare Testobjekte
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    """Erstellt eine frische CartPole-Umgebung und schließt sie nach dem Test."""
    e = gym.make("CartPole-v1")
    yield e
    e.close()

@pytest.fixture
def model():
    """Kleines Netz (hidden=32) damit Tests schnell laufen."""
    return ActorCritic(input_size=4, hidden_size=32, num_actions=2)

@pytest.fixture
def agent(env):
    """Agent mit minimalem Training (wenige Episoden) für Integrationstests."""
    return A2CAgentCartPole(env, num_episodes=5, max_steps=50, hidden_size=32)


# ══════════════════════════════════════════════════════════════════════════════
#  WHITEBOX-TESTS 
# ══════════════════════════════════════════════════════════════════════════════

class TestActorCriticNetwork:
    """Prüft das Netzwerk auf korrekte Ausgabeformen und -eigenschaften."""

    def test_output_shapes_single_state(self, model):
        """
        Whitebox: Für einen einzelnen Zustand (kein Batch) müssen
        action_probs die Form [num_actions] und value die Form [1] haben.
        """
        state = torch.FloatTensor([0.1, -0.2, 0.05, 0.3])
        action_probs, value = model(state)

        assert action_probs.shape == (2,), \
            f"Erwartet (2,), bekommen {action_probs.shape}"
        assert value.shape == (1,), \
            f"Erwartet (1,), bekommen {value.shape}"

    def test_output_shapes_batched(self, model):
        """
        Whitebox: Mit einem Batch von 8 Zuständen muss action_probs
        die Form [8, 2] und value die Form [8, 1] haben.
        """
        batch = torch.FloatTensor(np.random.rand(8, 4))
        action_probs, value = model(batch)

        assert action_probs.shape == (8, 2)
        assert value.shape == (8, 1)

    def test_action_probs_sum_to_one(self, model):
        """
        Whitebox: Softmax-Ausgabe des Actors muss sich zu 1 summieren
        (Wahrscheinlichkeitsverteilung).
        """
        state = torch.FloatTensor([0.1, -0.2, 0.05, 0.3])
        action_probs, _ = model(state)

        assert torch.isclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6), \
            f"Summe der Wahrscheinlichkeiten: {action_probs.sum().item():.6f}, erwartet 1.0"

    def test_action_probs_non_negative(self, model):
        """
        Whitebox: Alle Aktionswahrscheinlichkeiten müssen ≥ 0 sein.
        Negative Werte würden auf einen fehlerhaften Aktivierungspfad hinweisen.
        """
        state = torch.FloatTensor([0.1, -0.2, 0.05, 0.3])
        action_probs, _ = model(state)

        assert (action_probs >= 0).all(), \
            "Mindestens eine Aktionswahrscheinlichkeit ist negativ"

    def test_value_is_scalar(self, model):
        """
        Whitebox: Der Critic-Kopf soll einen einzelnen skalaren Wert V(s)
        ausgeben, keinen Vektor.
        """
        state = torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
        _, value = model(state)

        assert value.numel() == 1, \
            f"Critic soll 1 Wert ausgeben, gibt aber {value.numel()} aus"

    def test_different_inputs_produce_different_outputs(self, model):
        """
        Whitebox: Zwei verschiedene Zustände sollen unterschiedliche
        Action-Probs erzeugen (Netz ist nicht konstant).
        """
        s1 = torch.FloatTensor([1.0, 0.0, 0.0, 0.0])
        s2 = torch.FloatTensor([0.0, 1.0, 0.0, 0.0])

        p1, _ = model(s1)
        p2, _ = model(s2)

        assert not torch.allclose(p1, p2), \
            "Verschiedene Eingaben erzeugen identische Ausgaben – Netz reagiert nicht auf Zustand"

    def test_gradients_flow_through_both_heads(self, model):
        """
        Whitebox: Nach einem Backward-Pass müssen Gradienten in BEIDEN
        Köpfen (actor_layer und critic_layer) vorhanden sein.
        Fehlen Gradienten, würde ein Kopf nicht lernen.
        """
        state = torch.FloatTensor([0.1, -0.2, 0.05, 0.3])
        action_probs, value = model(state)

        # Einfacher kombinierten Verlust konstruieren
        loss = -torch.log(action_probs[0]) + value.squeeze()
        loss.backward()

        assert model.actor_layer.weight.grad is not None, \
            "Kein Gradient im Actor-Layer"
        assert model.critic_layer.weight.grad is not None, \
            "Kein Gradient im Critic-Layer"
        assert model.hidden_layer.weight.grad is not None, \
            "Kein Gradient in der gemeinsamen Hidden-Layer"


class TestComputeReturns:
    """Prüft die Berechnung der diskontierten Returns direkt."""

    def test_single_reward(self, agent):
        """
        Whitebox: Bei nur einem Reward r muss G_0 = r sein (kein Discount nötig).
        """
        returns = agent.compute_returns([1.0])
        assert torch.isclose(returns[0], torch.tensor(1.0)), \
            f"Erwartet 1.0, bekommen {returns[0].item()}"

    def test_two_rewards_manual(self, agent):
        """
        Whitebox: G_0 = r_0 + γ·r_1 manuell nachgerechnet.
        Mit gamma=0.99, r=[1.0, 1.0]:
            G_1 = 1.0
            G_0 = 1.0 + 0.99 * 1.0 = 1.99
        """
        returns = agent.compute_returns([1.0, 1.0])

        expected_G1 = 1.0
        expected_G0 = 1.0 + agent.gamma * 1.0

        assert torch.isclose(returns[1], torch.tensor(expected_G1), atol=1e-5)
        assert torch.isclose(returns[0], torch.tensor(expected_G0), atol=1e-5), \
            f"Erwartet {expected_G0}, bekommen {returns[0].item()}"

    def test_returns_length_matches_rewards(self, agent):
        """
        Whitebox: Die Länge der Returns muss der Länge der Rewards entsprechen.
        """
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
        returns = agent.compute_returns(rewards)

        assert len(returns) == len(rewards), \
            f"Länge stimmt nicht überein: {len(returns)} vs {len(rewards)}"

    def test_future_rewards_discounted_less(self, agent):
        """
        Whitebox: Weiter entfernte Belohnungen sollen weniger zum Return
        beitragen als nähere (Kernprinzip des Discountings).
        G_0 bei [0,0,0,1] muss kleiner sein als bei [1,0,0,0].
        """
        # Belohnung nur am Ende
        returns_late  = agent.compute_returns([0.0, 0.0, 0.0, 1.0])
        # Belohnung nur am Anfang
        returns_early = agent.compute_returns([1.0, 0.0, 0.0, 0.0])

        assert returns_late[0] < returns_early[0], \
            "Späte Belohnung sollte weniger wert sein als frühe"

    def test_zero_rewards(self, agent):
        """
        Whitebox: Bei allen Rewards = 0 müssen alle Returns ebenfalls 0 sein.
        """
        returns = agent.compute_returns([0.0, 0.0, 0.0])
        assert torch.all(returns == 0.0), \
            f"Erwartet nur Nullen, bekommen: {returns}"

    def test_returns_tensor_on_correct_device(self, agent):
        """
        Whitebox: Die Returns müssen auf demselben Gerät liegen wie das Netz,
        sonst schlägt der Loss-Backprop fehl.
        """
        returns = agent.compute_returns([1.0, 1.0])
        assert returns.device.type == agent.device.type


class TestTrainingLoop:
    """Prüft interne Zustände während und nach dem Training."""

    def test_parameters_change_after_training(self, env):
        """
        Whitebox: Nach dem Training müssen sich die Netzwerkparameter
        verändert haben. Wenn nicht, findet kein Lernen statt.
        """
        agent = A2CAgentCartPole(env, num_episodes=10, max_steps=50, hidden_size=32)

        # Gewichte vor dem Training kopieren
        params_before = [p.clone() for p in agent.policy_net.parameters()]

        agent.train()

        params_after = list(agent.policy_net.parameters())
        changed = any(
            not torch.equal(b, a)
            for b, a in zip(params_before, params_after)
        )
        assert changed, "Netzwerkparameter haben sich nach dem Training nicht verändert"

    def test_train_returns_correct_length(self, env):
        """
        Whitebox: Der Rückgabewert von train() muss genau num_episodes
        Einträge enthalten.
        """
        num_episodes = 7
        agent = A2CAgentCartPole(env, num_episodes=num_episodes, max_steps=50, hidden_size=32)
        rewards = agent.train()

        assert len(rewards) == num_episodes, \
            f"Erwartet {num_episodes} Rewards, bekommen {len(rewards)}"


# ══════════════════════════════════════════════════════════════════════════════
#  BLACKBOX-TESTS  – nur Input/Output, keine Interna
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentBehavior:
    """Prüft das beobachtbare Verhalten des Agenten von außen."""

    def test_action_is_valid(self, agent, env):
        """
        Blackbox: Der Agent muss immer eine gültige Aktion zurückgeben
        (0 = links, 1 = rechts bei CartPole).
        Wir testen das über mehrere Zustände hinweg.
        """
        state, _ = env.reset()
        for _ in range(20):
            action_probs, _ = agent.policy_net(torch.FloatTensor(state))
            action = torch.multinomial(action_probs, 1).item()

            assert action in [0, 1], \
                f"Ungültige Aktion: {action}, erwartet 0 oder 1"

            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()

    def test_train_returns_numpy_array(self, env):
        """
        Blackbox: train() soll ein numpy-Array zurückgeben,
        damit es direkt für Plots verwendet werden kann.
        """
        agent = A2CAgentCartPole(env, num_episodes=5, max_steps=50, hidden_size=32)
        rewards = agent.train()

        assert isinstance(rewards, np.ndarray), \
            f"Erwartet np.ndarray, bekommen {type(rewards)}"

    def test_episode_rewards_are_positive(self, env):
        """
        Blackbox: CartPole gibt pro Schritt +1 Belohnung, also muss
        jede Episode mindestens 1 Schritt (= Reward ≥ 1) dauern.
        """
        agent = A2CAgentCartPole(env, num_episodes=10, max_steps=50, hidden_size=32)
        rewards = agent.train()

        assert (rewards >= 1).all(), \
            f"Mindestens eine Episode hat Reward < 1: {rewards}"

    def test_reward_improves_over_long_training(self):
        """
        Blackbox: Ein längeres Training soll im Schnitt besser werden.
        Wir vergleichen den Ø-Reward der ersten 100 vs. der letzten 100 Episoden.
        Dieser Test ist probabilistisch – er kann selten fehlschlagen.
        """
        train_env = gym.make("CartPole-v1")
        agent = A2CAgentCartPole(
            train_env, num_episodes=500, max_steps=500, hidden_size=128
        )
        rewards = agent.train()

        early_mean = rewards[:100].mean()
        late_mean  = rewards[-100:].mean()

        assert late_mean > early_mean, (
            f"Kein Lernfortschritt: früher Ø={early_mean:.1f}, "
            f"später Ø={late_mean:.1f}"
        )

    def test_action_probs_sum_to_one_on_env_state(self, agent, env):
        """
        Blackbox: Auch auf einem echten Umgebungszustand muss die
        Summe der Aktionswahrscheinlichkeiten 1 ergeben.
        """
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state)
        action_probs, _ = agent.policy_net(state_tensor)

        assert torch.isclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5), \
            f"Summe der Probs: {action_probs.sum().item()}"

