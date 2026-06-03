import random
import math
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.s_head = nn.Linear(hidden_dim, state_dim)
        self.r_head = nn.Linear(hidden_dim, 1)
        self.p_head = nn.Linear(hidden_dim, 1)

    def forward(self, s_t, a_t):
        x = torch.cat([s_t, a_t], dim=-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.s_head(x), self.r_head(x), self.p_head(x)

    def loss(self, s_t, a_t, s_t1, r_t, p_t):
        s_hat, r_hat, p_logits = self.forward(s_t, a_t)
        state_loss = F.mse_loss(s_hat, s_t1)
        reward_loss = F.mse_loss(r_hat, r_t)
        p_loss = F.binary_cross_entropy_with_logits(p_logits, p_t)
        return state_loss + reward_loss + p_loss


def collect_transitions(env, n_episodes=200):
    buffer = []
    n_actions = env.action_space.n

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False

        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            a_onehot = np.zeros(n_actions, dtype=np.float32)
            a_onehot[a] = 1.0

            buffer.append((
                s.astype(np.float32),
                a_onehot,
                s_next.astype(np.float32),
                np.array([r], dtype=np.float32),
                np.array([float(not terminated)], dtype=np.float32)
            ))
            s = s_next

    return buffer


def train_dynamics(n_training_steps=5000, batch_size=64, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    model = DynamicsModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print("Sammle Transitionen fuer Dynamics Model...")
    buffer = collect_transitions(env, n_episodes=200)
    print(f"Buffer: {len(buffer)} Transitionen\n")

    losses = []
    model.train()
    for step in range(n_training_steps):
        idx = np.random.randint(0, len(buffer), size=batch_size)
        s, a, s_next, r, p = zip(*[buffer[i] for i in idx])

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        s_next = torch.tensor(np.array(s_next), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        p = torch.tensor(np.array(p), dtype=torch.float32)

        opt.zero_grad()
        loss = model.loss(s, a, s_next, r, p)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if step % 500 == 0:
            print(f"  Dynamics Step {step:5d} | Loss: {loss.item():.6f}")

    env.close()
    model.eval()
    print("Dynamics Model trainiert.\n")
    return model, np.array(losses)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-BASED ENV
# ══════════════════════════════════════════════════════════════════════════════

class ModelBasedEnv:
    MAX_STEPS = 500

    def __init__(self, dynamics, real_env):
        self.dynamics = dynamics
        self.real_env = real_env
        self.observation_space = real_env.observation_space
        self.action_space = real_env.action_space
        self._n_actions = real_env.action_space.n
        self._obs = None
        self._steps = 0

    def reset(self):
        self._obs, info = self.real_env.reset()
        self._steps = 0
        return self._obs, info

    def step(self, action):
        s_t = torch.FloatTensor(self._obs).unsqueeze(0)
        a_onehot = torch.zeros(1, self._n_actions)
        a_onehot[0, action] = 1.0

        with torch.no_grad():
            s_hat, r_hat, p_logits = self.dynamics(s_t, a_onehot)

        next_obs = s_hat.squeeze(0).numpy()
        reward = r_hat.item()

        self._steps += 1
        self._obs = next_obs

        p = torch.sigmoid(p_logits).item()
        terminated = p < 0.5
        truncated = self._steps >= self.MAX_STEPS

        return next_obs, reward, terminated, truncated, {}

    def close(self):
        self.real_env.close()


# ══════════════════════════════════════════════════════════════════════════════
#  A2C AGENT
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.actor_layer = nn.Linear(hidden_size, num_actions)
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return F.softmax(self.actor_layer(x), dim=-1), self.critic_layer(x)


class A2CAgentCartPole:
    def __init__(self, env, num_episodes=500, max_steps=500, gamma=0.99, lr=3e-4, hidden_size=128):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.policy_net = ActorCritic(input_size, hidden_size, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_loss = nn.MSELoss()

    def _state_to_tensor(self, state):
        return torch.FloatTensor(state).to(self.device)

    def compute_returns_bootstrap(self, rewards, last_state, done):
        if done:
            R = 0.0
        else:
            with torch.no_grad():
                last_tensor = self._state_to_tensor(last_state)
                _, bootstrap_value = self.policy_net(last_tensor)
            R = bootstrap_value.item()

        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self):
        episode_rewards = []
        avg100 = []
        losses = []

        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:
                state, _ = self.env.reset()
                episode_reward = 0.0
                values, rewards, log_probs = [], [], []
                last_state, done = state, False

                for _ in range(self.max_steps):
                    state_tensor = self._state_to_tensor(state)
                    action_probs, value = self.policy_net(state_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[action])

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    values.append(value)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    episode_reward += reward
                    last_state = next_state
                    state = next_state

                    if done:
                        break

                episode_rewards.append(episode_reward)
                returns = self.compute_returns_bootstrap(rewards, last_state, done)
                values = torch.cat(values)
                log_probs = torch.stack(log_probs)
                advantage = returns - values.detach()

                actor_loss = -(log_probs * advantage).mean()
                critic_loss = self.critic_loss(values, returns)
                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                losses.append(total_loss.item())
                avg100.append(np.mean(episode_rewards[-100:]))

                pbar.set_description(
                    f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | Ø100: {avg100[-1]:6.2f}"
                )

        return np.array(episode_rewards), np.array(avg100), np.array(losses)


def evaluate_agent(agent, env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            s = torch.FloatTensor(obs)
            with torch.no_grad():
                probs, _ = agent.policy_net(s)
            action = probs.argmax().item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISIERUNG
# ══════════════════════════════════════════════════════════════════════════════

def get_font(size=22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_cartpole_state_on_overlay(state, size, color_rgba):
    w, h = size
    overlay = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    x, x_dot, theta, theta_dot = state

    track_y = int(h * 0.72)
    draw.line((30, track_y, w - 30, track_y), fill=(90, 90, 90, 120), width=4)

    world_half_width = 2.4
    cart_center_x = int((x / world_half_width) * (w * 0.32)) + w // 2
    cart_center_x = max(60, min(w - 60, cart_center_x))

    cart_w, cart_h = 90, 36
    cart_left = cart_center_x - cart_w // 2
    cart_top = track_y - cart_h - 10
    cart_right = cart_center_x + cart_w // 2
    cart_bottom = track_y - 10

    draw.rectangle(
        (cart_left, cart_top, cart_right, cart_bottom),
        fill=color_rgba,
        outline=(20, 20, 20, color_rgba[3]),
        width=2
    )

    pole_len = 120
    pivot_x = cart_center_x
    pivot_y = cart_top + 4

    end_x = pivot_x + pole_len * math.sin(theta)
    end_y = pivot_y - pole_len * math.cos(theta)

    draw.line((pivot_x, pivot_y, end_x, end_y), fill=color_rgba, width=8)
    draw.ellipse((pivot_x - 5, pivot_y - 5, pivot_x + 5, pivot_y + 5), fill=(0, 0, 0, 180))

    return overlay


def make_overlay_frame(pred_state, true_state, action, pred_reward, true_reward, pred_p, step_idx):
    canvas_w, canvas_h = 1200, 900
    img = Image.new("RGBA", (canvas_w, canvas_h), (245, 246, 250, 255))
    draw = ImageDraw.Draw(img)

    title_font = get_font(34)
    text_font = get_font(24)
    small_font = get_font(20)

    draw.text((40, 24), "CartPole Overlay: Prediction vs True", fill=(30, 30, 30), font=title_font)
    draw.text((40, 72), f"Step {step_idx} | Action {action}", fill=(80, 80, 80), font=text_font)

    panel = (60, 130, 1140, 520)
    draw.rounded_rectangle(panel, radius=18, outline=(160, 160, 160), width=2, fill=(255, 255, 255))

    panel_w = panel[2] - panel[0] - 40
    panel_h = panel[3] - panel[1] - 40

    pred_overlay = draw_cartpole_state_on_overlay(pred_state, (panel_w, panel_h), (230, 120, 20, 150))
    true_overlay = draw_cartpole_state_on_overlay(true_state, (panel_w, panel_h), (40, 100, 220, 150))

    merged = Image.new("RGBA", (panel_w, panel_h), (255, 255, 255, 0))
    merged.alpha_composite(true_overlay)
    merged.alpha_composite(pred_overlay)
    img.alpha_composite(merged, dest=(panel[0] + 20, panel[1] + 20))

    draw.rounded_rectangle((70, 540, 250, 585), radius=10, fill=(40, 100, 220, 190))
    draw.text((270, 548), "= True state", fill=(40, 40, 40), font=text_font)

    draw.rounded_rectangle((520, 540, 700, 585), radius=10, fill=(230, 120, 20, 190))
    draw.text((720, 548), "= Predicted state", fill=(40, 40, 40), font=text_font)

    table_y = 630
    row_h = 42

    draw.text((70, table_y), "Variable", fill=(20, 20, 20), font=text_font)
    draw.text((280, table_y), "True", fill=(20, 20, 20), font=text_font)
    draw.text((500, table_y), "Prediction", fill=(20, 20, 20), font=text_font)
    draw.text((790, table_y), "Abs. Error", fill=(20, 20, 20), font=text_font)

    labels = ["x", "x_dot", "theta", "theta_dot"]
    for i, label in enumerate(labels):
        yy = table_y + (i + 1) * row_h
        tv = float(true_state[i])
        pv = float(pred_state[i])
        err = abs(tv - pv)

        draw.text((70, yy), label, fill=(40, 40, 40), font=text_font)
        draw.text((280, yy), f"{tv:+.5f}", fill=(40, 100, 220), font=text_font)
        draw.text((500, yy), f"{pv:+.5f}", fill=(230, 120, 20), font=text_font)
        draw.text((790, yy), f"{err:.5f}", fill=(150, 40, 40), font=text_font)

    info_y = table_y + 5 * row_h + 18
    draw.text((70, info_y), f"True reward: {true_reward:.3f}", fill=(40, 100, 220), font=text_font)
    draw.text((380, info_y), f"Pred reward: {pred_reward:.3f}", fill=(230, 120, 20), font=text_font)
    draw.text((700, info_y), f"Pred survival p: {pred_p:.3f}", fill=(70, 70, 70), font=text_font)

    hint_y = info_y + 80
    draw.line((40, hint_y - 22, 1140, hint_y - 22), fill=(210, 210, 210), width=2)

    draw.text(
        (40, hint_y),
        "Blau = true, Orange = prediction; sichtbare Trennung = Modellfehler.",
        fill=(90, 90, 90),
        font=small_font
    )

    return np.array(img.convert("RGB"))


def generate_overlay_gif(model, out_path="cartpole_overlay_comparison.gif", max_steps=80, fps=2, seed=123):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)

    frames = []
    done = False
    step_idx = 0

    while not done and step_idx < max_steps:
        action = env.action_space.sample()

        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a_onehot = torch.zeros(1, env.action_space.n, dtype=torch.float32)
        a_onehot[0, action] = 1.0

        with torch.no_grad():
            pred_next, pred_reward, pred_p_logits = model(s_t, a_onehot)

        pred_next = pred_next.squeeze(0).cpu().numpy()
        pred_reward = pred_reward.item()
        pred_p = torch.sigmoid(pred_p_logits).item()

        true_next, true_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = make_overlay_frame(
            pred_state=pred_next,
            true_state=true_next,
            action=action,
            pred_reward=pred_reward,
            true_reward=true_reward,
            pred_p=pred_p,
            step_idx=step_idx + 1
        )
        frames.append(frame)

        obs = true_next
        step_idx += 1

    env.close()
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"GIF gespeichert unter: {out_path}")


def plot_training_progress(dyn_losses, episode_rewards, avg100, a2c_losses):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax[0].plot(dyn_losses, color="tab:blue", alpha=0.7)
    ax[0].set_title("CartPole Dynamics Model Loss")
    ax[0].set_xlabel("Training step")
    ax[0].set_ylabel("Loss")

    ax[1].plot(episode_rewards, alpha=0.3, label="Episode Reward")
    if len(avg100) > 0:
        ax[1].plot(avg100, label="100ep avg")
    ax[1].plot(np.arange(len(a2c_losses)), a2c_losses, color="tab:red", alpha=0.3, label="A2C Loss")
    ax[1].set_title("CartPole Learning Progress")
    ax[1].set_xlabel("Episode / step")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    fig.savefig("cartpole_learning_progress.png", dpi=180)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("SCHRITT 1: Dynamics Model Training")
    print("=" * 60)
    dynamics, dyn_losses = train_dynamics(
        n_training_steps=5000,
        batch_size=64,
        lr=1e-3,
        seed=42
    )

    print("=" * 60)
    print("SCHRITT 2: A2C Training (Model-Based)")
    print("=" * 60)
    real_env = gym.make("CartPole-v1")
    fake_env = ModelBasedEnv(dynamics, real_env)
    agent = A2CAgentCartPole(fake_env, num_episodes=500)
    episode_rewards, avg100, a2c_losses = agent.train()

    plot_training_progress(dyn_losses, episode_rewards, avg100, a2c_losses)

    print("\n" + "=" * 60)
    print("SCHRITT 3: Evaluation im echten Environment")
    print("=" * 60)
    eval_env = gym.make("CartPole-v1")
    mean_r, std_r = evaluate_agent(agent, eval_env, n_episodes=20)
    print(f"Reward ueber 20 echte Episoden: {mean_r:.1f} +/- {std_r:.1f}")
    eval_env.close()
    real_env.close()

    print("\n" + "=" * 60)
    print("SCHRITT 4: Overlay-GIF")
    print("=" * 60)
    generate_overlay_gif(
        dynamics,
        out_path="cartpole_overlay_comparison.gif",
        max_steps=80,
        fps=2,
        seed=123
    )

    print("Fertig.")