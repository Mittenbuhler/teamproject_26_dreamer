import numpy as np
import torch
from torch.utils.data import Dataset
import gymnasium as gym
from collections import deque
import random

ACTION_SIZE = 2
VISIBLE_STATE_INDICES = np.array([0, 2])


def visible_state(full_state):
    return np.array(full_state)[VISIBLE_STATE_INDICES].astype(np.float32)


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
    def __init__(self, episodes, seq_len):
        # Wir filtern extrem fehlerhafte/leere Episoden vorsichtshalber aus
        self.episodes = [ep for ep in episodes if len(ep["actions"]) >= 2]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        
        # Rohe Arrays aus der Episode extrahieren
        obs = np.array(ep["vis"], dtype=np.float32)        # Form: [ep_len + 1, obs_dim]
        acts = np.array(ep["actions"], dtype=np.float32)    # Form: [ep_len, act_dim]
        rews = np.array(ep["rewards"], dtype=np.float32)    # Form: [ep_len, 1]
        conts = np.array(ep["continues"], dtype=np.float32)  # Form: [ep_len, 1]
        
        ep_len = len(acts)
        
        # Bestimme, wie viel wir maximal herausschneiden können
        slice_len = min(ep_len, self.seq_len)
        
        # Slice die echten Daten bis zur maximal verfügbaren Länge
        obs_slice = obs[:slice_len + 1]
        acts_slice = acts[:slice_len]
        rews_slice = rews[:slice_len]
        conts_slice = conts[:slice_len]
        
        # Falls die Episode kürzer ist als seq_len, padden wir mit Nullen auf
        if slice_len < self.seq_len:
            pad_len = self.seq_len - slice_len
            
            # Padding für Observations (Zielform: [seq_len + 1, obs_dim])
            obs_pad = np.zeros((pad_len, obs.shape[-1]), dtype=np.float32)
            obs_slice = np.concatenate([obs_slice, obs_pad], axis=0)
            
            # Padding für Actions (Zielform: [seq_len, act_dim])
            acts_pad = np.zeros((pad_len, acts.shape[-1]), dtype=np.float32)
            acts_slice = np.concatenate([acts_slice, acts_pad], axis=0)
            
            # Padding für Rewards (Zielform: [seq_len, 1])
            rews_pad = np.zeros((pad_len, rews.shape[-1]), dtype=np.float32)
            rews_slice = np.concatenate([rews_slice, rews_pad], axis=0)
            
            # Padding für Continues (Zielform: [seq_len, 1])
            # Wichtig: Abgelaufene Schritte erhalten als Continue-Signal ohnehin eine 0.0
            conts_pad = np.zeros((pad_len, conts.shape[-1]), dtype=np.float32)
            conts_slice = np.concatenate([conts_slice, conts_pad], axis=0)
            
        return {
            "observations": torch.tensor(obs_slice),
            "actions": torch.tensor(acts_slice),
            "rewards": torch.tensor(rews_slice),
            "continues": torch.tensor(conts_slice)
        }


def collect_episodes(env, actor, n_episodes=10, max_steps=200, seed=None, epsilon=0.05):
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