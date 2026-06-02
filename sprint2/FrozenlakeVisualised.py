import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=64):
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
        target      = s_t1.argmax(dim=-1)
        state_loss  = F.cross_entropy(s_hat, target)
        reward_loss = F.mse_loss(r_hat, r_t)
        p_loss      = F.binary_cross_entropy_with_logits(p_logits, p_t)
        return state_loss + reward_loss + p_loss


def collect_transitions(env, n_episodes=200):
    buffer = []
    n_actions = env.action_space.n
    n_states  = env.observation_space.n
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a_onehot      = np.zeros(n_actions, dtype=np.float32)
            a_onehot[a]   = 1.0
            s_onehot      = np.eye(n_states, dtype=np.float32)[s]
            s_next_onehot = np.eye(n_states, dtype=np.float32)[s_next]
            buffer.append((
                s_onehot, a_onehot, s_next_onehot,
                np.array([r], dtype=np.float32),
                np.array([float(not terminated)], dtype=np.float32)
            ))
            s = s_next
    return buffer


def train_dynamics(n_training_steps=5000, batch_size=64, lr=1e-3):
    env   = gym.make("FrozenLake-v1", is_slippery=False)
    model = DynamicsModel()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    print("Sammle Transitionen fuer Dynamics Model...")
    buffer = collect_transitions(env, n_episodes=200)
    print(f"Buffer: {len(buffer)} Transitionen\n")
    model.train()
    for step in range(n_training_steps):
        idx = np.random.randint(0, len(buffer), size=batch_size)
        s, a, s_next, r, p = zip(*[buffer[i] for i in idx])
        s, a, s_next, r, p = (torch.tensor(np.array(x)) for x in (s, a, s_next, r, p))
        opt.zero_grad()
        loss = model.loss(s, a, s_next, r, p)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  Dynamics Step {step:5d} | Loss: {loss.item():.6f}")
    env.close()
    model.eval()
    print("Dynamics Model trainiert.\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-BASED ENV
# ══════════════════════════════════════════════════════════════════════════════

class ModelBasedEnv:
    MAX_STEPS = 500

    def __init__(self, dynamics, real_env):
        self.dynamics          = dynamics
        self.real_env          = real_env
        self.observation_space = real_env.observation_space
        self.action_space      = real_env.action_space
        self._n_actions        = real_env.action_space.n
        self._obs   = None
        self._steps = 0

    def reset(self):
        self._obs, info = self.real_env.reset()
        self._steps = 0
        return self._obs, info

    def step(self, action):
        n_states = self.real_env.observation_space.n
        s_t = torch.FloatTensor(np.eye(n_states)[int(self._obs)]).unsqueeze(0)
        a_onehot = torch.zeros(1, self._n_actions)
        a_onehot[0, action] = 1.0
        with torch.no_grad():
            s_hat, r_hat, p_logits = self.dynamics(s_t, a_onehot)
        next_obs = s_hat.squeeze(0).argmax().item()
        reward   = r_hat.item()
        self._steps += 1
        self._obs    = next_obs
        p          = torch.sigmoid(p_logits).item()
        terminated = p < 0.5
        truncated  = self._steps >= self.MAX_STEPS
        return next_obs, reward, terminated, truncated, {}

    def close(self):
        self.real_env.close()


# ══════════════════════════════════════════════════════════════════════════════
#  A2C AGENT
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, hidden_size, num_outputs):
        super().__init__()
        self.hidden_layer = nn.Linear(16, hidden_size)
        self.actor_layer  = nn.Linear(hidden_size, num_outputs)
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return F.softmax(self.actor_layer(x), dim=-1), self.critic_layer(x)


class A2CAgentFrozenLake:
    def __init__(self, env, num_episodes=500, max_steps=200, gamma=0.99, lr=1e-3, hidden_size=128):
        self.env          = env
        self.num_episodes = num_episodes
        self.max_steps    = max_steps
        self.gamma        = gamma
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net   = ActorCritic(hidden_size, env.action_space.n).to(self.device)
        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_loss  = nn.MSELoss()

    def compute_returns(self, rewards):
        R, returns = 0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)

    def train(self):
        episode_rewards = np.array([])
        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:
                state, _ = self.env.reset()
                values, rewards, logits = [], [], []
                episode_reward = 0
                for _ in range(self.max_steps):
                    s = torch.FloatTensor(np.eye(16)[state]).to(self.device)
                    action_probs, value = self.policy_net(s)
                    action = torch.multinomial(action_probs, 1).item()
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    values.append(value)
                    rewards.append(reward)
                    logits.append(torch.log(action_probs[action]))
                    episode_reward += reward
                    state = next_state
                    if done:
                        break
                episode_rewards = np.append(episode_rewards, episode_reward)
                returns   = self.compute_returns(rewards)
                values    = torch.cat(values)
                logits    = torch.stack(logits)
                advantage = returns - values
                loss = -(logits * advantage.detach()).mean() + self.critic_loss(values, returns)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    f"Episode {episode} | Reward (100ep avg): {np.mean(episode_rewards[-100:]):.3f}"
                )
        self.env.close()
        return episode_rewards




# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_agent(agent, env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            s = torch.FloatTensor(np.eye(16)[obs])
            with torch.no_grad():
                probs, _ = agent.policy_net(s)
            action = probs.argmax().item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    real_env = gym.make("FrozenLake-v1", is_slippery=False)

    print("=" * 60)
    print("SCHRITT 1: Dynamics Model Training")
    print("=" * 60)
    dynamics = train_dynamics(n_training_steps=5000, batch_size=64)

    print("=" * 60)
    print("SCHRITT 2: A2C Training (Model-Based)")
    print("=" * 60)
    fake_env = ModelBasedEnv(dynamics, real_env)
    agent    = A2CAgentFrozenLake(fake_env, num_episodes=500)
    agent.train()

    print("\n" + "=" * 60)
    print("SCHRITT 3: Evaluation im echten Environment")
    print("=" * 60)
    eval_env      = gym.make("FrozenLake-v1", is_slippery=False)
    mean_r, std_r = evaluate_agent(agent, eval_env, n_episodes=20)
    print(f"Reward ueber 20 echte Episoden: {mean_r:.1f} +/- {std_r:.1f}")
    eval_env.close()
    real_env.close()

    print("\n" + "=" * 60)
    