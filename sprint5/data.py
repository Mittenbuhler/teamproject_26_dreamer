import numpy as np
import torch
import gymnasium as gym
from collections import deque
import random
from PIL import Image

from .models import DEVICE, ACTION_SIZE

IMG_SIZE = 32


def onehot_action(a, action_size=ACTION_SIZE):
    v = np.zeros(action_size, dtype=np.float32)
    v[a] = 1.0
    return v


def render_frame(env):
    """Rendert den aktuellen Environment-Zustand als (1, IMG_SIZE, IMG_SIZE)
    Graustufen-Tensor, normalisiert auf [0, 1].

    Erfordert env = gym.make(..., render_mode="rgb_array")."""
    frame = env.render()  # (H, W, 3) uint8
    img = Image.fromarray(frame).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :]  # (1, IMG_SIZE, IMG_SIZE)


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
            vis = np.array(ep["vis"], dtype=np.float32)        # (L+1, 1, IMG_SIZE, IMG_SIZE)
            acts = np.array(ep["actions"], dtype=np.float32)    # (L, action_size)
            rews = np.array(ep["rewards"], dtype=np.float32)    # (L, 1)
            conts = np.array(ep["continues"], dtype=np.float32)  # (L, 1)
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
            # Fallback: falls keine Episode lang genug fuer seq_len war,
            # nimm die volle (kuerzere) Episode als einzige Sequenz.
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
            "observations": torch.from_numpy(o),  # (T+1, 1, IMG_SIZE, IMG_SIZE)
            "actions": torch.from_numpy(a),
            "rewards": torch.from_numpy(r),
            "continues": torch.from_numpy(c),
        }


@torch.no_grad()
def select_action(actor, flatz, h, epsilon=0.0):
    """Waehlt eine Aktion anhand der aktuellen RSSM-Features (flatz, h).
    Mit Wahrscheinlichkeit epsilon wird stattdessen zufaellig exploriert."""
    if random.random() < epsilon:
        a_idx = random.randrange(ACTION_SIZE)
    else:
        feat = torch.cat([flatz, h], dim=-1)
        logits = actor(feat)
        a_idx = torch.distributions.Categorical(logits=logits).sample().item()
    a = torch.tensor(onehot_action(a_idx), device=flatz.device, dtype=torch.float32).unsqueeze(0)
    return a, a_idx


def collect_episodes(env, world_model, actor, n_episodes=10, max_steps=200, seed=None, epsilon=0.05):
    """Sammelt Episoden im echten Environment. Der RSSM-State (flatz, h) wird
    online Schritt fuer Schritt fortgefuehrt (world_model.step_online), genau
    wie im Training in observe_forward -- es gibt keinen separaten
    Beobachtungs-Encoder-Shortcut mehr."""
    episodes = []
    actor.eval()
    world_model.eval()
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep_idx)
        ep = {"fulls": [], "vis": [], "actions": [], "rewards": [], "continues": []}
        flatz, h = world_model.initial(batch_size=1, device=DEVICE)

        frame = render_frame(env)  # Bild des Zustands direkt nach reset
        done = False
        steps = 0
        while not done and steps < max_steps:
            ep["fulls"].append(obs)
            ep["vis"].append(frame)

            a, a_idx = select_action(actor, flatz, h, epsilon=epsilon)
            next_full, r, terminated, truncated, _ = env.step(a_idx)
            done = terminated or truncated

            ep["actions"].append(onehot_action(a_idx))
            ep["rewards"].append([r])
            ep["continues"].append([0.0 if terminated else 1.0])

            frame = render_frame(env)
            next_vis = torch.tensor(frame, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flatz, h = world_model.step_online(flatz, h, a, next_vis)

            obs = next_full
            steps += 1

        ep["fulls"].append(obs)
        ep["vis"].append(frame)
        episodes.append(ep)
    return episodes