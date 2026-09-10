import numpy as np
import torch
from torch.utils.data import Dataset
import gymnasium as gym
from collections import deque
import random

from .models import DEVICE

ACTION_SIZE = 2
FULL_STATE_SIZE = 4  # CartPole: [x, x_dot, theta, theta_dot]
# Volle 4D-Sicht: das gesamte Dreamer-System (Weltmodell + Policy) arbeitet
# jetzt auf dem kompletten, markovschen CartPole-Zustand.
VISIBLE_STATE_INDICES = np.array([0, 1, 2, 3])


def visible_state(state):
    # 4D-Beobachtung fuer das WELTMODELL (alle Zustandsvariablen).
    return np.array(state, dtype=np.float32)


def full_state(state):
    # 4D-Beobachtung fuer die POLICY (identisch zu visible_state).
    return np.array(state, dtype=np.float32)


def onehot_action(a, action_size=ACTION_SIZE):
    v = np.zeros(action_size, dtype=np.float32)
    v[a] = 1.0
    return v


class ReplayBuffer:
    def __init__(self, capacity=300):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, ep):
        self.buffer.append(ep)

    def __len__(self):
        return len(self.buffer)

    def sample_episode(self):
        return random.choice(self.buffer)

    def sample_episodes(self, n):
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)


class SequenceDataset(Dataset):
    """Zerlegt Episoden in Fenster der Laenge seq_len.

    Wichtig: Fuer Episoden, die mindestens seq_len Schritte haben, werden
    ECHTE, ueberlappungsfreie Fenster erzeugt (kein Null-Padding). Nur fuer
    kuerzere Episoden faellt das Dataset auf das alte Padding-Verhalten
    zurueck. Das verhindert, dass Null-Padding die Reward-/Continue-Signale
    verwaessert -- was sonst das Weltmodell (und damit den Actor) verzerrt.
    """

    def __init__(self, episodes, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.episodes = [ep for ep in episodes if len(ep["actions"]) >= 2]
        # Index-Liste aus (episode, start). Lange Episoden -> mehrere Fenster.
        self.index = []
        for ep in self.episodes:
            T = len(ep["actions"])
            if T >= seq_len:
                last = T - seq_len
                for start in range(0, last + 1, self.stride):
                    self.index.append((ep, start, False))  # echtes Fenster
            else:
                self.index.append((ep, 0, True))           # Padding-Fallback

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, start, needs_pad = self.index[idx]
        obs = np.array(ep["vis"], dtype=np.float32)
        acts = np.array(ep["actions"], dtype=np.float32)
        rews = np.array(ep["rewards"], dtype=np.float32)
        conts = np.array(ep["continues"], dtype=np.float32)

        if not needs_pad:
            s, L = start, self.seq_len
            obs_slice = obs[s:s + L + 1]
            acts_slice = acts[s:s + L]
            rews_slice = rews[s:s + L]
            conts_slice = conts[s:s + L]
            mask_slice = np.ones((L, 1), dtype=np.float32)
        else:
            ep_len = len(acts)
            slice_len = min(ep_len, self.seq_len)
            obs_slice = obs[:slice_len + 1]
            acts_slice = acts[:slice_len]
            rews_slice = rews[:slice_len]
            conts_slice = conts[:slice_len]
            pad_len = self.seq_len - slice_len
            obs_slice = np.concatenate([obs_slice, np.zeros((pad_len, obs.shape[-1]), dtype=np.float32)], axis=0)
            acts_slice = np.concatenate([acts_slice, np.zeros((pad_len, acts.shape[-1]), dtype=np.float32)], axis=0)
            rews_slice = np.concatenate([rews_slice, np.zeros((pad_len, rews.shape[-1]), dtype=np.float32)], axis=0)
            conts_slice = np.concatenate([conts_slice, np.zeros((pad_len, conts.shape[-1]), dtype=np.float32)], axis=0)
            # Maske: echte Schritte = 1, Padding = 0
            mask_slice = np.concatenate([
                np.ones((slice_len, 1), dtype=np.float32),
                np.zeros((pad_len, 1), dtype=np.float32),
            ], axis=0)

        return {
            "observations": torch.tensor(obs_slice),
            "actions": torch.tensor(acts_slice),
            "rewards": torch.tensor(rews_slice),
            "continues": torch.tensor(conts_slice),
            "mask": torch.tensor(mask_slice),
        }


def _act_in_env(env, world_model, actor, n_episodes, max_steps, seed, epsilon):
    """Kanonischer DreamerV2-Handel: fuehrt den RSSM-Zustand ueber die Episode
    mit. Die Policy (actor) operiert auf den Latent-Features [flatz, h] --
    dieselbe Repraesentation wie im Imagination-Training."""
    episodes = []
    world_model.eval()
    actor.eval()
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep_idx)
        ep = {"fulls": [], "vis": [], "actions": [], "rewards": [], "continues": []}

        # RSSM-Startzustand (Nullzustand) + Null-Aktion.
        flatz, h = world_model.initial(batch_size=1, device=DEVICE)
        prev_action = torch.zeros(1, world_model.action_size, device=DEVICE)

        # Ersten Zustand ins RSSM geben (erster act_step mit Start-Obs).
        obs_t = torch.tensor(visible_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)

        done = False
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                logits = actor(feat)
                if random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = torch.distributions.Categorical(logits=logits).sample().item()

            next_full, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ep["fulls"].append(obs)
            ep["vis"].append(visible_state(obs))
            ep["actions"].append(onehot_action(a))
            ep["rewards"].append([r])
            ep["continues"].append([0.0 if done else 1.0])

            # RSSM-Zustand mit der ausgefuehrten Aktion + neuer Beobachtung fortschreiben.
            prev_action = torch.tensor(onehot_action(a), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            next_obs_t = torch.tensor(visible_state(next_full), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flatz, h, feat = world_model.act_step(flatz, h, prev_action, next_obs_t)

            obs = next_full
            steps += 1

        ep["fulls"].append(obs)
        ep["vis"].append(visible_state(obs))
        episodes.append(ep)
    return episodes


def collect_episodes(env, actor, world_model=None, n_episodes=10, max_steps=200, seed=None, epsilon=0.05):
    """Sammelt Episoden.

    - Wird world_model uebergeben, laeuft der kanonische DreamerV2-Handel: die
      Policy (actor) handelt auf den RSSM-Latent-Features. Das ist der korrekte
      Pfad fuer echtes Lernen.
    - Ohne world_model faellt die Funktion auf den alten Direkt-Pfad zurueck
      (actor bekommt die Beobachtung direkt), fuer Rueckwaertskompatibilitaet.
    """
    if world_model is not None:
        return _act_in_env(env, world_model, actor, n_episodes, max_steps, seed, epsilon)

    episodes = []
    actor.eval()
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep_idx)
        ep = {"fulls": [], "vis": [], "actions": [], "rewards": [], "continues": []}
        done = False
        steps = 0
        while not done and steps < max_steps:
            x = torch.tensor(visible_state(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = actor(x)
                if random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = torch.distributions.Categorical(logits=logits).sample().item()
            next_full, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep["fulls"].append(obs)
            ep["vis"].append(visible_state(obs))
            ep["actions"].append(onehot_action(a))
            ep["rewards"].append([r])
            ep["continues"].append([0.0 if done else 1.0])
            obs = next_full
            steps += 1
        ep["fulls"].append(obs)
        ep["vis"].append(visible_state(obs))
        episodes.append(ep)
    return episodes