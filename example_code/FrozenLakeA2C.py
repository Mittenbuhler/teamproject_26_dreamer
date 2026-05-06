import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio
from IPython.display import Image, display


# ── Netzwerk ──────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size):
        super().__init__()
        self.embedding   = nn.Embedding(num_states, hidden_size)
        self.hidden      = nn.Linear(hidden_size, hidden_size)
        self.actor_head  = nn.Linear(hidden_size, num_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.embedding(x))
        x = F.relu(self.hidden(x))
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        value        = self.critic_head(x)
        return action_probs, value


# ── Agent ─────────────────────────────────────────────────────────────────────

class A2CAgentFrozenLake:
    def __init__(self,
                 num_episodes=5000,
                 max_steps=100,
                 gamma=0.99,
                 lr=1e-3,
                 hidden_size=128,
                 entropy_coef=0.01):

        self.env = gym.make("FrozenLake-v1", is_slippery=False)
        self.num_episodes  = num_episodes
        self.max_steps     = max_steps
        self.gamma         = gamma
        self.entropy_coef  = entropy_coef
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_states  = self.env.observation_space.n
        num_actions = self.env.action_space.n

        self.net            = ActorCritic(num_states, num_actions, hidden_size).to(self.device)
        self.optimizer      = optim.Adam(self.net.parameters(), lr=lr)
        self.critic_loss_fn = nn.MSELoss()

    def choose_action(self, state):
        s = torch.tensor([state], dtype=torch.long).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.net(s)
        return torch.multinomial(action_probs, 1).item()

    def compute_returns(self, rewards):
        R, returns = 0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self):
        episode_rewards = []

        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:
                state, _ = self.env.reset()
                rewards, values, log_probs, entropies = [], [], [], []
                episode_reward = 0

                for _ in range(self.max_steps):
                    s = torch.tensor([state], dtype=torch.long).to(self.device)
                    action_probs, value = self.net(s)

                    action   = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[0, action])
                    entropy  = -(action_probs * torch.log(action_probs + 1e-8)).sum()

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                episode_rewards.append(episode_reward)

                returns   = self.compute_returns(rewards)
                values    = torch.cat(values).squeeze()
                log_probs = torch.stack(log_probs)
                entropies = torch.stack(entropies)
                advantage = returns - values.detach()

                actor_loss   = -(log_probs * advantage).mean()
                critic_loss  = self.critic_loss_fn(values, returns)
                entropy_loss = -entropies.mean()
                loss         = actor_loss + critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if episode % 100 == 0:
                    avg = np.mean(episode_rewards[-100:])
                    pbar.set_description(f"Episode {episode} | Avg Reward: {avg:.3f}")

        self.env.close()
        return np.array(episode_rewards)


# ── Training starten ──────────────────────────────────────────────────────────

agent = A2CAgentFrozenLake(
    num_episodes=5000,
    hidden_size=128,
    lr=1e-3,
    entropy_coef=0.01
)
rewards = agent.train()
print(f"Durchschnittlicher Reward (letzte 100): {np.mean(rewards[-100:]):.3f}")


# ── GIF aufnehmen ─────────────────────────────────────────────────────────────

try:
    render_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    use_legacy_render = False
except TypeError:
    render_env = gym.make("FrozenLake-v1", is_slippery=False)
    use_legacy_render = True

def unpack_reset(reset_result):
    if isinstance(reset_result, tuple):
        observation, info = reset_result
        return observation
    return reset_result

def unpack_step(step_result):
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        observation, reward, done, info = step_result
    return observation, reward, done

def render_frame(env, use_legacy_render):
    if use_legacy_render:
        return env.render(mode="rgb_array")
    return env.render()

def record_episode_frozenlake(agent, env, use_legacy_render, output_path="a2c_frozenlake.gif"):
    frames = []
    observation = unpack_reset(env.reset())
    episode_reward = 0
    done = False

    while not done:
        frames.append(render_frame(env, use_legacy_render))

        state_tensor = torch.tensor([observation], dtype=torch.long).to(agent.device)

        with torch.no_grad():
            action_probs, _ = agent.net(state_tensor)

        action = action_probs.argmax(dim=1).item()
        observation, reward, done = unpack_step(env.step(action))
        episode_reward += reward

    frames.append(render_frame(env, use_legacy_render))

    imageio.mimsave(output_path, frames, fps=4)
    return output_path, episode_reward


gif_path, episode_reward = record_episode_frozenlake(
    agent, render_env, use_legacy_render
)
render_env.close()

print(f"Episode Reward: {episode_reward}")
display(Image(filename=gif_path))