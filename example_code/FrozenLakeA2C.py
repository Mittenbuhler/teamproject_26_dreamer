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
    def __init__(self, hidden_size, num_outputs):
        super(ActorCritic, self).__init__()
        self.hidden_layer = nn.Linear(16, hidden_size)
        self.actor_layer  = nn.Linear(hidden_size, num_outputs)
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        action_probs = F.softmax(self.actor_layer(x), dim=-1)
        value        = self.critic_layer(x)
        return action_probs, value


# ── Agent ─────────────────────────────────────────────────────────────────────

class A2CAgentFrozenLake:
    def __init__(self, env, num_episodes=5000, max_steps=100, gamma=0.99, lr=1e-3, hidden_size=128):
        self.env          = env
        self.num_episodes = num_episodes
        self.max_steps    = max_steps
        self.gamma        = gamma
        self.lr           = lr
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net   = ActorCritic(hidden_size, env.action_space.n).to(self.device)
        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_loss  = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(np.eye(16)[state]).to(self.device)
        action_probs, _ = self.policy_net(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def compute_returns(self, rewards):
        R, returns = 0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        return returns

    def train(self):
        episode_rewards = np.array([])
        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:
                state, _ = self.env.reset()
                episode_reward = 0
                values  = []
                rewards = []
                logits  = []

                for step in range(self.max_steps):
                    state = torch.FloatTensor(np.eye(16)[state]).to(self.device)
                    action_probs, value = self.policy_net(state)
                    action = torch.multinomial(action_probs, 1).item()
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    new_probs = torch.log(action_probs[action])
                    values.append(value)
                    rewards.append(reward)
                    logits.append(new_probs)
                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                episode_rewards = np.append(episode_rewards, episode_reward)

                returns   = self.compute_returns(rewards)
                values    = torch.cat(values)
                logits    = torch.stack(logits)
                advantage = returns - values

                actorLoss  = -(logits * advantage.detach()).mean()
                criticLoss = self.critic_loss(values, returns)
                lossEval   = actorLoss + criticLoss

                self.optimizer.zero_grad()
                lossEval.backward()
                self.optimizer.step()

                pbar.set_description(f"Episode {episode}, Reward: {np.mean(episode_rewards[-100:]):.3f}")

        self.env.close()
        return episode_rewards


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

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

def record_episode(agent, env, use_legacy_render, output_path="a2c_frozenlake.gif"):
    frames = []
    observation    = unpack_reset(env.reset())
    episode_reward = 0
    done           = False
    device         = next(agent.policy_net.parameters()).device

    while not done:
        frames.append(render_frame(env, use_legacy_render))
        observation_tensor = torch.FloatTensor(np.eye(16)[observation]).to(device)
        with torch.no_grad():
            action_probs, _ = agent.policy_net(observation_tensor)
        action = action_probs.argmax(dim=0).item()
        observation, reward, done = unpack_step(env.step(action))
        episode_reward += reward

    frames.append(render_frame(env, use_legacy_render))
    imageio.mimsave(output_path, frames, fps=4)
    return output_path, episode_reward


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Training
    env = gym.make("FrozenLake-v1", is_slippery=False)
    a2c_model = A2CAgentFrozenLake(env)
    a2c_model.train()

    # GIF aufnehmen
    try:
        render_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
        use_legacy_render = False
    except TypeError:
        render_env = gym.make("FrozenLake-v1", is_slippery=False)
        use_legacy_render = True

    gif_path, episode_reward = record_episode(a2c_model, render_env, use_legacy_render)
    render_env.close()

    print(f"Episode Reward: {episode_reward}")
    display(Image(filename=gif_path))