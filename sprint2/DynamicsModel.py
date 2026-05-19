import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import numpy as np


class DynamicsModel(nn.Module):
  def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
    super(DynamicsModel, self).__init__()
    self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 6x64
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)      
    self.s_head = nn.Linear(hidden_dim, state_dim)   
    self.r_head = nn.Linear(hidden_dim, 1)            
  
  def forward(self,s_t,a_t):
    """
        s_t: Tensor [batch, 4]   – aktueller State
        a_t: Tensor [batch, 2]   – Action als One-Hot
        Gibt zurück: s_hat_t+1 [batch, 4], r_hat_t [batch, 1]
        """
    x = torch.cat([s_t, a_t], dim=-1)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    s_hat_next = self.s_head(x)
    r_hat = self.r_head(x)
    return s_hat_next, r_hat

  def loss(self, s_t, a_t, s_t1, r_t):
        """
        s_t:   aktueller State     [batch, 4]
        a_t:   Action (One-Hot)    [batch, 2]
        s_t1:  nächster State      [batch, 4]
        r_t:   Reward              [batch, 1]
        """
        s_hat_next, r_hat = self.forward(s_t, a_t)

        state_loss  = F.mse_loss(s_hat_next, s_t1)
        reward_loss = F.mse_loss(r_hat, r_t)

        return state_loss + reward_loss


def collect_transitions(env, n_episodes=200):
    """Sammelt zufällige Erfahrungen (s_t, a_t, s_t+1, r_t)."""
    buffer = []
    n_actions = env.action_space.n

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()                    # zufällige Aktion
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Action als One-Hot kodieren
            a_onehot = np.zeros(n_actions, dtype=np.float32)
            a_onehot[a] = 1.0

            buffer.append((
                s.astype(np.float32),
                a_onehot,
                s_next.astype(np.float32),
                np.array([r], dtype=np.float32),
            ))
            s = s_next

    return buffer


def train(n_training_steps=5000, batch_size=64, lr=1e-3):
    env = gym.make("CartPole-v1")
    model = DynamicsModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Sammle Transitionen...")
    buffer = collect_transitions(env, n_episodes=200)
    print(f"Buffer-Größe: {len(buffer)} Transitionen")

    for step in range(n_training_steps):
        # Zufälliger Mini-Batch aus dem Buffer
        indices = np.random.randint(0, len(buffer), size=batch_size)
        s, a, s_next, r = zip(*[buffer[i] for i in indices])

        s      = torch.tensor(np.array(s))
        a      = torch.tensor(np.array(a))
        s_next = torch.tensor(np.array(s_next))
        r      = torch.tensor(np.array(r))

        optimizer.zero_grad()
        l = model.loss(s, a, s_next, r)
        l.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:5d} | Loss: {l.item():.6f}")

    env.close()
    return model

#Methode um das Model zu bewerten
def evaluate_dynamics_model(model, buffer, n_samples=5):
    """Zeigt Vorhersagen vs. echte Werte an."""
    indices = np.random.randint(0, len(buffer), size=n_samples)
    
    print(f"{'':=<70}")
    print(f"Dynamics Model Evaluation ({n_samples} Samples)")
    print(f"{'':=<70}")
    
    total_mse_state  = 0
    total_mse_reward = 0

    for i in indices:
        s, a, s_next, r = buffer[i]

        s_t    = torch.tensor(s).unsqueeze(0)      # [1, 4]
        a_t    = torch.tensor(a).unsqueeze(0)      # [1, 2]
        s_next = torch.tensor(s_next).unsqueeze(0) # [1, 4]
        r_t    = torch.tensor(r).unsqueeze(0)      # [1, 1]

        with torch.no_grad():
            s_hat, r_hat = model.forward(s_t, a_t)

        mse_state  = F.mse_loss(s_hat, s_next).item()
        mse_reward = F.mse_loss(r_hat, r_t).item()
        total_mse_state  += mse_state
        total_mse_reward += mse_reward

        print(f"\nSample {i}:")
        print(f"  State     | Echt: {s_next.numpy()[0]}  |  Pred: {s_hat.numpy()[0].round(3)}")
        print(f"  Reward    | Echt: {r_t.item():.3f}              |  Pred: {r_hat.item():.3f}")
        print(f"  MSE State: {mse_state:.6f}  |  MSE Reward: {mse_reward:.6f}")

    print(f"\n{'':=<70}")
    print(f"Durchschnittlicher MSE State:  {total_mse_state  / n_samples:.6f}")
    print(f"Durchschnittlicher MSE Reward: {total_mse_reward / n_samples:.6f}")
    print(f"{'':=<70}")

    # ── Hauptprogramm ─────────────────────────────────────────────────────────────

model = train()  
buffer = collect_transitions(gym.make("CartPole-v1"), n_episodes=10)
evaluate_dynamics_model(model, buffer, n_samples=5)