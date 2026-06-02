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
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
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
        """
        s_t:   aktueller State     [batch, 4]
        a_t:   Action (One-Hot)    [batch, 2]
        s_t1:  nächster State      [batch, 4]
        r_t:   Reward              [batch, 1]
        p_t:   Done-Target         [batch, 1]
        """
        #generate s_t+1,r_t
        s_hat, r_hat, p_logits = self.forward(s_t, a_t)
        
        #MSE
        state_loss  = F.mse_loss(s_hat, s_t1)
        reward_loss = F.mse_loss(r_hat, r_t)
        
        # Binary Cross Entropy mit Logits für die binäre Klassifikation (Done)
        p_loss   = F.binary_cross_entropy_with_logits(p_logits, p_t)
        
        return state_loss + reward_loss + p_loss


def collect_transitions(env, n_episodes=200):
    """Sammelt zufällige Erfahrungen (s_t, a_t, s_t+1, r_t)."""
    buffer = []
    n_actions = env.action_space.n

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            #zufällige Aktion
            a = env.action_space.sample()
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Action als One-Hot kodieren
            a_onehot = np.zeros(n_actions, dtype=np.float32)
            a_onehot[a] = 1.0

            # Wir speichern 'not terminated' als p_target (1.0 = läuft stabil, 0.0 = umgefallen)
            buffer.append((
                s.astype(np.float32), 
                a_onehot,
                s_next.astype(np.float32), 
                np.array([r], dtype=np.float32),
                np.array([float(not terminated)], dtype=np.float32)
            ))
            s = s_next

    return buffer


def train_dynamics(n_training_steps=5000, batch_size=64, lr=1e-3):
    #initialize env, model, optimizer
    env   = gym.make("CartPole-v1")
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
        s_t = torch.FloatTensor(self._obs).unsqueeze(0)      # [1, 4]
        a_onehot = torch.zeros(1, self._n_actions)
        a_onehot[0, action] = 1.0                            # [1, 2]

        with torch.no_grad():
            # Wir fangen das p_logits-Signal aus dem Modell ab
            s_hat, r_hat, p_logits = self.dynamics(s_t, a_onehot)

        next_obs = s_hat.squeeze(0).numpy()                  # (4,)
        reward   = r_hat.item()

        self._steps += 1
        self._obs    = next_obs

        # Statt physikalischen Grenzen berechnen wir die Fortlauf-Wahrscheinlichkeit p via Sigmoid.
        # Für p < 0.5 entscheidet das Modell: Der Stab ist umgefallen!
        p = torch.sigmoid(p_logits).item()
        terminated = p < 0.5
        
        truncated = self._steps >= self.MAX_STEPS

        return next_obs, reward, terminated, truncated, {}


# ══════════════════════════════════════════════════════════════════════════════
#  A2C AGENT  (unverändert aus CartPoleA2C.py)
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.actor_layer  = nn.Linear(hidden_size, num_actions)
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return F.softmax(self.actor_layer(x), dim=-1), self.critic_layer(x)


class A2CAgentCartPole:
    def __init__(self, env, num_episodes=500, max_steps=500, gamma=0.99, lr=3e-4, hidden_size=128):
        self.env          = env
        self.num_episodes = num_episodes
        self.max_steps    = max_steps
        self.gamma        = gamma
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size  = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.policy_net  = ActorCritic(input_size, hidden_size, num_actions).to(self.device)
        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=lr)
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
        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:
                state, _ = self.env.reset()
                episode_reward = 0.0
                values, rewards, log_probs = [], [], []
                last_state, done = state, False   

                for _ in range(self.max_steps):
                    state_tensor = self._state_to_tensor(state)
                    action_probs, value = self.policy_net(state_tensor)
                    action   = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[action])

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    values.append(value)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    episode_reward += reward
                    last_state = next_state   
                    state      = next_state
                    if done:
                        break

                episode_rewards.append(episode_reward)
                returns   = self.compute_returns_bootstrap(rewards, last_state, done)
                values    = torch.cat(values)
                log_probs = torch.stack(log_probs)
                advantage = returns - values.detach()

                actor_loss  = -(log_probs * advantage).mean()
                critic_loss = self.critic_loss(values, returns)
                total_loss  = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                mean_reward = np.mean(episode_rewards[-100:])
                pbar.set_description(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:6.1f} | "
                    f"Ø100: {mean_reward:6.2f}"
                )
        return np.array(episode_rewards)


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
#  Hauptprogramm
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    real_env = gym.make("CartPole-v1")

    print("=" * 60)
    print("SCHRITT 1: Dynamics Model Training")
    print("=" * 60)
    dynamics = train_dynamics(n_training_steps=5000, batch_size=64)

    print("=" * 60)
    print("SCHRITT 2: A2C Training (Model-Based)")
    print("=" * 60)
    fake_env = ModelBasedEnv(dynamics, real_env)
    agent    = A2CAgentCartPole(fake_env, num_episodes=500)
    agent.train()

    print("\n" + "=" * 60)
    print("SCHRITT 3: Evaluation im echten Environment")
    print("=" * 60)
    eval_env       = gym.make("CartPole-v1")
    mean_r, std_r  = evaluate_agent(agent, eval_env, n_episodes=20)
    print(f"Reward über 20 echte Episoden:  {mean_r:.1f} ± {std_r:.1f}")
    eval_env.close()
    real_env.close()