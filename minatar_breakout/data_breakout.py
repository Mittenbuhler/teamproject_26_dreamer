"""
data_breakout.py — Datensammlung fuer MinAtar Breakout.

Basiert auf data.py (CartPole). Geaenderte Stellen mit # [BREAKOUT] markiert.
Nutzt die NATIVE MinAtar-API (minatar.Environment), weil der Gymnasium-Wrapper
versionsabhaengig oft nicht registriert ist.

Konventionen:
  - State: MinAtar liefert (10,10,4) bool -> wir permutieren zu (4,10,10) float32
    (PyTorch (C,H,W)) und speichern das in "vis".
  - Aktionen: minimal_action_set() gibt IDs wie [0,1,3]; wir mappen sie auf
    Indizes [0,1,2] und speichern one-hot der Laenge 3.
  - Reward: 0 oder 1 direkt aus der Umgebung.
"""
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from minatar import Environment              # [BREAKOUT] native API
from .models_breakout import DEVICE, OBS_CHANNELS, OBS_HW, ACTION_SIZE


def make_env(game="breakout", sticky_action_prob=0.1, seed=None):
    # [BREAKOUT] native MinAtar-Umgebung. Der Seed wird (falls gewuenscht) global
    # ueber numpy gesetzt, da der Konstruktor keinen Seed-Parameter hat.
    if seed is not None:
        np.random.seed(seed)
    env = Environment(game, sticky_action_prob=sticky_action_prob)
    return env


def state_to_obs(state):
    # [BREAKOUT] (10,10,4) bool  ->  (4,10,10) float32
    return np.asarray(state, dtype=np.float32).transpose(2, 0, 1)


def onehot_action(a_idx, action_size=ACTION_SIZE):
    v = np.zeros(action_size, dtype=np.float32)
    v[a_idx] = 1.0
    return v


class ReplayBuffer:
    def __init__(self, capacity=2000):
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
        return random.sample(list(self.buffer), min(n, len(self.buffer)))


class SequenceDataset(Dataset):
    """Gleitende Fenster wie bei CartPole. vis ist jetzt (4,10,10) pro Schritt."""
    def __init__(self, episodes, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.episodes = [ep for ep in episodes if len(ep["actions"]) >= 2]
        self.index = []
        for ep in self.episodes:
            T = len(ep["actions"])
            if T >= seq_len:
                for start in range(0, T - seq_len + 1, self.stride):
                    self.index.append((ep, start, False))
            else:
                self.index.append((ep, 0, True))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, start, needs_pad = self.index[idx]
        vis = np.array(ep["vis"], dtype=np.float32)        # (T+1, 4,10,10)
        acts = np.array(ep["actions"], dtype=np.float32)   # (T, 3)
        rews = np.array(ep["rewards"], dtype=np.float32)   # (T, 1)
        conts = np.array(ep["continues"], dtype=np.float32)

        if not needs_pad:
            s, L = start, self.seq_len
            obs_slice = vis[s:s + L + 1]
            acts_slice = acts[s:s + L]
            rews_slice = rews[s:s + L]
            conts_slice = conts[s:s + L]
            mask_slice = np.ones((L, 1), dtype=np.float32)
        else:
            L = self.seq_len
            sl = min(len(acts), L)
            obs_slice = vis[:sl + 1]
            acts_slice = acts[:sl]; rews_slice = rews[:sl]; conts_slice = conts[:sl]
            pad = L - sl
            # [BREAKOUT] Obs-Padding mit Null-Bildern (4,10,10)
            obs_slice = np.concatenate([obs_slice, np.zeros((pad, OBS_CHANNELS, OBS_HW, OBS_HW), dtype=np.float32)], axis=0)
            acts_slice = np.concatenate([acts_slice, np.zeros((pad, ACTION_SIZE), dtype=np.float32)], axis=0)
            rews_slice = np.concatenate([rews_slice, np.zeros((pad, 1), dtype=np.float32)], axis=0)
            conts_slice = np.concatenate([conts_slice, np.zeros((pad, 1), dtype=np.float32)], axis=0)
            mask_slice = np.concatenate([np.ones((sl, 1), dtype=np.float32), np.zeros((pad, 1), dtype=np.float32)], axis=0)

        return {
            "observations": torch.tensor(obs_slice),
            "actions": torch.tensor(acts_slice),
            "rewards": torch.tensor(rews_slice),
            "continues": torch.tensor(conts_slice),
            "mask": torch.tensor(mask_slice),
        }


@torch.no_grad()
def collect_episodes(env_unused, actor, world_model, n_episodes=10, max_steps=1000,
                     seed=None, epsilon=0.05, game="breakout"):
    """[BREAKOUT] Kanonischer RSSM-Handel auf MinAtar Breakout.

    Hinweis: env_unused bleibt fuer Signatur-Kompatibilitaet; wir erstellen die
    native MinAtar-Umgebung hier selbst, da sie nicht wiederverwendbar resettet
    wie Gym. actor + world_model handeln auf den RSSM-Latent-Features.
    """
    world_model.eval(); actor.eval()
    episodes = []
    for ep_idx in range(n_episodes):
        env = make_env(game, seed=None if seed is None else seed + ep_idx)
        action_map = env.minimal_action_set()      # z.B. [0,1,3] -> Index 0,1,2
        env.reset()

        ep = {"vis": [], "actions": [], "rewards": [], "continues": []}
        flatz, h = world_model.initial(batch_size=1, device=DEVICE)
        prev_action = torch.zeros(1, ACTION_SIZE, device=DEVICE)

        obs_np = state_to_obs(env.state())
        obs_t = torch.tensor(obs_np, device=DEVICE).unsqueeze(0)   # (1,4,10,10)
        flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)

        done = False; steps = 0
        while not done and steps < max_steps:
            if random.random() < epsilon:
                a_idx = random.randrange(ACTION_SIZE)
            else:
                logits = actor(feat)
                a_idx = torch.distributions.Categorical(logits=logits).sample().item()

            reward, done = env.act(action_map[a_idx])   # [BREAKOUT] ID-Mapping

            ep["vis"].append(obs_np)
            ep["actions"].append(onehot_action(a_idx))
            ep["rewards"].append([float(reward)])
            ep["continues"].append([0.0 if done else 1.0])

            obs_np = state_to_obs(env.state())
            prev_action = torch.tensor(onehot_action(a_idx), device=DEVICE).unsqueeze(0)
            obs_t = torch.tensor(obs_np, device=DEVICE).unsqueeze(0)
            flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)
            steps += 1

        ep["vis"].append(obs_np)   # letzter Zustand -> vis hat Laenge T+1
        episodes.append(ep)
    return episodes