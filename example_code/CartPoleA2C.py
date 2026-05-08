import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio


# ══════════════════════════════════════════════════════════════════════════════
#  NETZWERK – ActorCritic
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        """
        input_size  : Anzahl der Zustandsmerkmale (4 bei CartPole)
        hidden_size : Größe der gemeinsam genutzten versteckten Schicht
        num_actions : Anzahl möglicher Aktionen (2 bei CartPole: links/rechts)
        """
        super(ActorCritic, self).__init__()

        # Gemeinsame Merkmalsextraktion – beide Köpfe profitieren davon
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        # Actor-Kopf: gibt Logits für jede Aktion aus
        self.actor_layer  = nn.Linear(hidden_size, num_actions)

        # Critic-Kopf: gibt einen skalaren Zustandswert V(s) aus
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Gemeinsame Aktivierung mit ReLU
        x = F.relu(self.hidden_layer(x))

        # Actor: Softmax wandelt Logits in Wahrscheinlichkeiten um
        action_probs = F.softmax(self.actor_layer(x), dim=-1)

        # Critic: linearer Ausgang → skalarer Wert V(s)
        value = self.critic_layer(x)

        return action_probs, value


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT – A2CAgentCartPole
# ══════════════════════════════════════════════════════════════════════════════

class A2CAgentCartPole:
    def __init__(
        self,
        env,
        num_episodes=1500,
        max_steps=500,      # CartPole-v1 endet spätestens nach 500 Schritten
        gamma=0.99,         # Diskontierungsfaktor: wie stark werden zukünftige Belohnungen gewichtet?
        lr=3e-4,
        hidden_size=128,
    ):
        self.env          = env
        self.num_episodes = num_episodes
        self.max_steps    = max_steps
        self.gamma        = gamma

        # Gerätewahl: GPU wenn verfügbar, sonst CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CartPole: 4 kontinuierliche Zustandsmerkmale, 2 diskrete Aktionen
        input_size  = env.observation_space.shape[0]   # = 4
        num_actions = env.action_space.n               # = 2

        self.policy_net  = ActorCritic(input_size, hidden_size, num_actions).to(self.device)
        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_loss = nn.MSELoss()

    # ── Hilfsmethode: Zustand → Tensor ──────────────────────────────────────

    def _state_to_tensor(self, state):
        """Wandelt den numpy-Array-Zustand in einen Float-Tensor um."""
        return torch.FloatTensor(state).to(self.device)

    # ── Hilfsmethode: Diskontierte Returns berechnen ─────────────────────────

    def compute_returns(self, rewards):
        """
        Berechnet die diskontierten kumulierten Belohnungen G_t für jeden Zeitschritt t:

            G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … = r_t + γ·G_{t+1}

        Wir iterieren rückwärts durch die Rewards, um dies effizient zu tun.

        γ nahe 1  → Agent denkt langfristig, zukünftige Belohnungen
                             zählen fast genauso viel wie sofortige.
        γ nahe 0          → Agent ist kurzfristig orientiert.
        """
        R, returns = 0.0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    # ── Trainingsschleife ────────────────────────────────────────────────────

    def train(self):
        """
        Der A2C-Algorithmus im Überblick (pro Episode):

        1. Episode abspielen: Zustände, Aktionen, Belohnungen und
           Critic-Schätzungen sammeln.

        2. Diskontierte Returns G_t berechnen.

        3. Advantage berechnen:
               A_t = G_t − V(s_t)
           Der Advantage misst, ob eine Aktion *besser oder schlechter*
           war als der Critic erwartet hatte.
           A_t > 0 → Aktion war überraschend gut  → Actor soll sie öfter wählen
           A_t < 0 → Aktion war enttäuschend      → Actor soll sie meiden

        4. Actor-Loss (Policy Gradient):
               L_actor = −log π(a_t|s_t) · A_t
           Das Minus macht aus Maximierung (Gradient Ascent) ein Minimierungsproblem.
           Detach() beim Advantage verhindert, dass Gradienten durch den Critic fließen.

        5. Critic-Loss (Value Function):
               L_critic = MSE(V(s_t), G_t)
           Der Critic soll besser lernen, den tatsächlichen Return vorherzusagen.

        6. Gesamtverlust = L_actor + L_critic → Backpropagation → Parameterupdate.
        """
        episode_rewards = []

        with tqdm(range(self.num_episodes)) as pbar:
            for episode in pbar:

                # ── Episode initialisieren ───────────────────────────────────
                # Gymnasium >= 0.26: reset() gibt (observation, info) zurück
                state, _ = self.env.reset()
                episode_reward = 0.0

                # Puffer für einen vollständigen Durchlauf
                values  = []   # Critic-Schätzungen V(s_t)
                rewards = []   # Erhaltene Belohnungen r_t
                log_probs = [] # Log-Wahrscheinlichkeiten der gewählten Aktionen

                # ── Einen Episodendurchlauf sammeln ──────────────────────────
                for _ in range(self.max_steps):
                    state_tensor = self._state_to_tensor(state)

                    # Forward Pass: Actor gibt π(a|s), Critic gibt V(s)
                    action_probs, value = self.policy_net(state_tensor)

                    # Aktion stochastisch sampeln (Exploration!)
                    # multinomial zieht eine Aktion proportional zu den Wahrscheinlichkeiten
                    action = torch.multinomial(action_probs, 1).item()

                    # Log-Wahrscheinlichkeit der gewählten Aktion speichern
                    # (wird für den Actor-Loss gebraucht)
                    log_prob = torch.log(action_probs[action])

                    # Gymnasium >= 0.26: step() gibt 5 Werte zurück
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    values.append(value)
                    rewards.append(reward)
                    log_probs.append(log_prob)

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                episode_rewards.append(episode_reward)

                # ── Verlust berechnen und Netz updaten ──────────────────────

                # Diskontierte Returns G_t für jeden Schritt der Episode
                returns = self.compute_returns(rewards)

                # Tensoren zusammenführen
                values    = torch.cat(values)          # Shape: [T]
                log_probs = torch.stack(log_probs)     # Shape: [T]

                # Advantage: wie viel besser/schlechter war die Aktion als erwartet?
                # detach() → Gradient fließt hier NICHT in den Critic zurück;
                # der Advantage dient dem Actor nur als skalarer Gewichtungsfaktor.
                advantage = returns - values.detach()

                # Actor-Loss: schlechte Aktionen bestrafen, gute verstärken
                actor_loss  = -(log_probs * advantage).mean()

                # Critic-Loss: Critic soll echte Returns besser vorhersagen
                critic_loss = self.critic_loss(values, returns)

                # Gesamtverlust 
                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Fortschrittsanzeige: gleitender Mittelwert der letzten 100 Episoden
                mean_reward = np.mean(episode_rewards[-100:])
                pbar.set_description(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:6.1f} | "
                    f"Ø100: {mean_reward:6.2f}"
                )

        self.env.close()
        return np.array(episode_rewards)


# ══════════════════════════════════════════════════════════════════════════════
#  GIF-AUFNAHME
# ══════════════════════════════════════════════════════════════════════════════

def record_episode(agent, env, output_path="a2c_cartpole.gif"):
    """
    Zeichnet eine Episode des trainierten Agenten auf und speichert sie als GIF.
    Im Gegensatz zum Training wird hier deterministisch gehandelt (argmax statt
    sampeln), um das erlernte Verhalten klar zu zeigen.
    """
    frames = []
    state, _ = env.reset()   # Modernes Gymnasium-API
    episode_reward = 0.0
    done = False
    device = next(agent.policy_net.parameters()).device

    while not done:
        frames.append(env.render())

        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action_probs, _ = agent.policy_net(state_tensor)

        # Deterministisch: wähle die Aktion mit höchster Wahrscheinlichkeit
        action = action_probs.argmax(dim=0).item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    frames.append(env.render())
    imageio.mimsave(output_path, frames, fps=30)
    print(f"GIF gespeichert: {output_path}")
    print(f"Episode-Belohnung: {episode_reward}")
    return output_path, episode_reward


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Training ──────────────────────────────────────────────────────────────
    print("=== Training startet ===")
    train_env = gym.make("CartPole-v1")
    agent = A2CAgentCartPole(train_env)
    rewards = agent.train()
    print(f"\nTraining abgeschlossen. Bester Reward: {rewards.max():.0f} | "
          f"Ø letzte 100 Episoden: {rewards[-100:].mean():.2f}")

    # ── Aufnahme ──────────────────────────────────────────────────────────────
    print("\n=== Zeichne Episode auf ===")
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    record_episode(agent, render_env)
    render_env.close()