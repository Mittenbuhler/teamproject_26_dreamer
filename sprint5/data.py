import numpy as np
import torch
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


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, seq_len=10):
        self.seqs = []
        for ep in episodes:
            vis = np.array(ep["vis"], dtype=np.float32)
            acts = np.array(ep["actions"], dtype=np.float32)
            rews = np.array(ep["rewards"], dtype=np.float32)
            conts = np.array(ep["continues"], dtype=np.float32)
            L = len(acts)
            if L < 1:
                continue
            start = 0
            while start + seq_len <= L:
                self.seqs.append(
                    (
                        vis[start:start + seq_len + 1],
                        acts[start:start + seq_len],
                        rews[start:start + seq_len],
                        conts[start:start + seq_len],
                    )
                )
                start += 1
        if len(self.seqs) == 0:
            for ep in episodes:
                vis = np.array(ep["vis"], dtype=np.float32)
                acts = np.array(ep["actions"], dtype=np.float32)
                rews = np.array(ep["rewards"], dtype=np.float32)
                conts = np.array(ep["continues"], dtype=np.float32)
                L = len(acts)
                if L < 1:
                    continue
                self.seqs.append((vis[:L + 1], acts[:L], rews[:L], conts[:L]))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        o, a, r, c = self.seqs[idx]
        return {
            "observations": torch.from_numpy(o),
            "actions": torch.from_numpy(a),
            "rewards": torch.from_numpy(r),
            "continues": torch.from_numpy(c),
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