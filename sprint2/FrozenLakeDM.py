import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS MODEL  (mit p als Überlebenswahrscheinlichkeit via Cross Entropy)
# ══════════════════════════════════════════════════════════════════════════════

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 6x64
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.s_head = nn.Linear(hidden_dim, state_dim)
        self.r_head = nn.Linear(hidden_dim, 1)
        # Ein dritter Kopf für die Done-Vorhersage (gibt unnormierte Logits aus)
        self.p_head = nn.Linear(hidden_dim, 1)

    def forward(self, s_t, a_t):
        """
        s_t: Tensor [batch, 4]   – aktueller State
        a_t: Tensor [batch, 2]   – Action als One-Hot
        Gibt zurück: s_hat_t+1 [batch, 4], r_hat_t [batch, 1], p_logits [batch, 1]
        """
        x = torch.cat([s_t, a_t], dim=-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        #s_t+1
        s_hat_next = self.s_head(x)
        #r_t
        r_hat = self.r_head(x)
        # Done-Prediction
        p_logits = self.p_head(x)
        return s_hat_next, r_hat, p_logits

    def loss(self, s_t, a_t, s_t1, r_t, p_t):
         s_hat, r_hat, p_logits = self.forward(s_t, a_t)
    
    # Cross-Entropy statt MSE für diskrete Zustände
         target = s_t1.argmax(dim=-1)          # One-Hot → Index
         state_loss  = F.cross_entropy(s_hat, target)
         reward_loss = F.mse_loss(r_hat, r_t)
         p_loss      = F.binary_cross_entropy_with_logits(p_logits, p_t)
    
         return state_loss + reward_loss + p_loss


def collect_transitions(env, n_episodes=2000):
    buffer = []
    n_actions = env.action_space.n
    n_states = env.observation_space.n  # 16 for FrozenLake

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            a_onehot = np.zeros(n_actions, dtype=np.float32)
            a_onehot[a] = 1.0

            # One-hot encode the discrete states
            s_onehot      = np.eye(n_states, dtype=np.float32)[s]
            s_next_onehot = np.eye(n_states, dtype=np.float32)[s_next]

            buffer.append((
                s_onehot,
                a_onehot,
                s_next_onehot,
                np.array([r], dtype=np.float32),
                np.array([float(not terminated)], dtype=np.float32)
            ))
            s = s_next

    return buffer


def train_dynamics(n_training_steps=5000, batch_size=64, lr=1e-3):
    #initialize env, model, optimizer
    env   = gym.make("FrozenLake-v1",is_slippery=False)
    model = DynamicsModel()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    #collect transitions
    print("Sammle Transitionen für Dynamics Model...")
    buffer = collect_transitions(env, n_episodes=200)
    print(f"Buffer: {len(buffer)} Transitionen\n")

    model.train()
    for step in range(n_training_steps):
        # Zufälliger Mini-Batch aus dem Buffer
        idx = np.random.randint(0, len(buffer), size=batch_size)
        s, a, s_next, r, p = zip(*[buffer[i] for i in idx])
        s, a, s_next, r, p = (torch.tensor(np.array(x)) for x in (s, a, s_next, r, p))

        #calcutlate loss
        opt.zero_grad()
        # Done-Vektor in die Loss-Berechnung hineingeben
        loss = model.loss(s, a, s_next, r, p)
        #update model parameters
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

    def __init__(self, dynamics: DynamicsModel, real_env: gym.Env):
        self.dynamics        = dynamics
        self.real_env        = real_env          # nur für reset() genutzt
        self.observation_space = real_env.observation_space
        self.action_space      = real_env.action_space
        self._n_actions        = real_env.action_space.n

        self._obs   = None
        self._steps = 0

    def reset(self):
        self._obs, info = self.real_env.reset()
        self._steps     = 0
        return self._obs, info

    def step(self, action: int):
        n_states = self.real_env.observation_space.n
        s_t = torch.FloatTensor(np.eye(n_states)[int(self._obs)]).unsqueeze(0)
        a_onehot = torch.zeros(1, self._n_actions)
        a_onehot[0, action] = 1.0

        with torch.no_grad():
            s_hat, r_hat, p_logits = self.dynamics(s_t, a_onehot)

        # Convert predicted one-hot back to discrete index
        next_obs = s_hat.squeeze(0).argmax().item()
        reward   = r_hat.item()

        self._steps += 1
        self._obs    = next_obs

        p = torch.sigmoid(p_logits).item()
        terminated = p < 0.5
        truncated  = self._steps >= self.MAX_STEPS

        return next_obs, reward, terminated, truncated, {}
    
    def close(self):
       self.real_env.close()


# ══════════════════════════════════════════════════════════════════════════════
#  A2C AGENT  (unverändert aus FrozenLakeA2C.py)
# ══════════════════════════════════════════════════════════════════════════════

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
    def __init__(self, env, num_episodes=5000, max_steps=200, gamma=0.99, lr=1e-3, hidden_size=128):
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

def evaluate_agent(agent, env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            s = torch.FloatTensor(np.eye(16)[obs])  # one-hot encode
            with torch.no_grad():
                probs, _ = agent.policy_net(s)
            action = probs.argmax().item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

# ══════════════════════════════════════════════════════════════════════════════
#  Hauptprogramm
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    real_env = gym.make("FrozenLake-v1",is_slippery=False)

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
    eval_env       = gym.make("FrozenLake-v1",is_slippery=False)
    mean_r, std_r  = evaluate_agent(agent, eval_env, n_episodes=20)
    print(f"Reward über 20 echte Episoden:  {mean_r:.1f} ± {std_r:.1f}")
    eval_env.close()
    real_env.close()