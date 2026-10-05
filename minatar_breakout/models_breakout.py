"""
models_breakout.py — RSSM-Variante fuer MinAtar Breakout (10x10x4 Bild-Beobachtungen).

Basiert auf models.py (CartPole, 4D-Vektor). Geaenderte Stellen sind mit
# [BREAKOUT] markiert. Der Behavior-Learning-Teil (imagine, prediction_heads
fuer reward/continue) bleibt UNVERAENDERT -- nur das Weltmodell-Frontend
(wie Beobachtungen rein- und rauskommen) wird auf Bilder umgestellt.

Konvention: intern (B, C, H, W) = (B, 4, 10, 10). MinAtar liefert (H, W, C);
die Permutation passiert beim Einlesen in data.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 4
K = 8
STOCHASTIC_SIZE = C * K
DETERMINISTIC_SIZE = 256   # [BREAKOUT] groesser als CartPole (64): Breakout ist komplexer;
                           # mehr Kapazitaet fuer die visuelle Dynamik (Hebel 4)
HIDDEN_SIZE = 300
# [BREAKOUT] Beobachtung ist jetzt ein Bild, keine Zahl mehr:
OBS_CHANNELS = 4           # paddle, ball, trail, brick
OBS_HW = 10                # 10x10 Grid
EMBED_SIZE = 128           # Encoder-Output
ACTION_SIZE = 3            # [BREAKOUT] links, rechts, no-op (v1-Actionset)


# [BREAKOUT] --- CNN-Encoder: 10x10x4 -> Embed-Vektor ---
class MinAtarEncoder(nn.Module):
    def __init__(self, channels=OBS_CHANNELS, embed_size=EMBED_SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),  # (16,10,10)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),        # (32,5,5)
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * 5 * 5, embed_size)

    def forward(self, x):
        # x: (B, 4, 10, 10)
        h = self.conv(x).reshape(x.shape[0], -1)
        return F.relu(self.fc(h))  # (B, embed_size)


# [BREAKOUT] --- CNN-Decoder: Latent-Feature -> 10x10x4 Logits ---
class MinAtarDecoder(nn.Module):
    def __init__(self, feat_size, channels=OBS_CHANNELS):
        super().__init__()
        self.fc = nn.Linear(feat_size, 32 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (16,10,10)
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 3, stride=1, padding=1),              # (4,10,10)
        )

    def forward(self, feat):
        h = self.fc(feat).reshape(-1, 32, 5, 5)
        return self.deconv(h)  # (B, 4, 10, 10) Logits


class RSSM(nn.Module):
    def __init__(self, categorical_size=C, class_size=K, stoch_size=STOCHASTIC_SIZE,
                 deter_size=DETERMINISTIC_SIZE, hidden_size=HIDDEN_SIZE,
                 action_size=ACTION_SIZE):
        super().__init__()
        self.C = categorical_size
        self.K = class_size
        self.stoch_size = stoch_size
        self.h_size = deter_size
        self.hid = hidden_size
        self.action_size = action_size

        self.action_stack = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.h_size),
        )
        self.gru = nn.GRUCell(self.h_size, self.h_size)

        self.prior_model = nn.Sequential(
            nn.Linear(self.h_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.C * self.K),
        )
        # [BREAKOUT] Posterior nimmt jetzt h + ENCODER-EMBED statt h + obs-Vektor
        self.encoder = MinAtarEncoder()
        self.posterior_model = nn.Sequential(
            nn.Linear(self.h_size + EMBED_SIZE, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.C * self.K),
        )

        self.pred_model = nn.Sequential(
            nn.Linear(self.stoch_size + self.h_size, self.hid),
            nn.ReLU(),
        )
        # [BREAKOUT] observation_head ist jetzt ein DECODER (CNN), kein Linear
        self.observation_decoder = MinAtarDecoder(feat_size=self.stoch_size + self.h_size)
        self.reward_head = nn.Linear(self.hid, 1)
        self.continue_head = nn.Linear(self.hid, 1)

    def initial(self, batch_size=1, device=None):
        dev = device or DEVICE
        return (torch.zeros(batch_size, self.stoch_size, device=dev),
                torch.zeros(batch_size, self.h_size, device=dev))

    def logits_to_shape(self, logits):
        return logits.view(logits.shape[0], self.C, self.K)

    def flatten_latent(self, z):
        return z.view(z.shape[0], self.C * self.K)

    def sample_straight_through(self, logits):
        probs = F.softmax(logits, dim=-1)
        B, Cc, Kk = probs.shape
        idx = torch.multinomial(probs.view(B * Cc, Kk), num_samples=1).squeeze(-1)
        onehot = F.one_hot(idx, num_classes=Kk).float().view(B, Cc, Kk)
        return onehot.detach() - probs.detach() + probs

    def mode_one_hot(self, logits):
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.K).float().to(logits.device)

    # [BREAKOUT] prediction_heads: observation kommt jetzt aus dem Decoder.
    # reward/continue bleiben identisch zu CartPole.
    def prediction_heads(self, flatz, h):
        feat = torch.cat([flatz, h], dim=-1)
        pred_feat = self.pred_model(feat)
        return {
            "observation": self.observation_decoder(feat),   # (B,4,10,10) Logits
            "reward": self.reward_head(pred_feat),
            "continuelogit": self.continue_head(pred_feat),
        }

    def observe_forward(self, observations, actions):
        # [BREAKOUT] observations: (B, T+1, 4, 10, 10) statt (B, T+1, 4)
        B, T = observations.shape[0], actions.shape[1]
        flatz, h = self.initial(batch_size=B, device=observations.device)

        prior_logits_list, posterior_logits_list = [], []
        prior_preds, posterior_preds = [], []

        for t in range(T):
            act_t = actions[:, t, :]
            h = self.gru(self.action_stack(torch.cat([flatz, act_t], dim=-1)), h)

            prior_logits = self.logits_to_shape(self.prior_model(h))
            prior_z = self.sample_straight_through(prior_logits)
            prior_preds.append(self.prediction_heads(self.flatten_latent(prior_z), h))

            # [BREAKOUT] naechste Beobachtung durch den Encoder jagen
            next_obs = observations[:, t + 1]                 # (B,4,10,10)
            embed = self.encoder(next_obs)                    # (B,EMBED_SIZE)
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, embed], dim=-1))
            )
            posterior_z = self.sample_straight_through(posterior_logits)
            posterior_flatz = self.flatten_latent(posterior_z)
            posterior_preds.append(self.prediction_heads(posterior_flatz, h))

            flatz = posterior_flatz
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

        def stack_dicts(dl):
            return {k: torch.stack([d[k] for d in dl], dim=1) for k in dl[0].keys()}

        return {
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),
            "prior_predictions": stack_dicts(prior_preds),
            "posterior_predictions": stack_dicts(posterior_preds),
        }

    @torch.no_grad()
    def posterior_start_state(self, observations, actions, t0=0):
        flatz, h = self.initial(batch_size=observations.shape[0], device=observations.device)
        for t in range(t0 + 1):
            act_t = actions[:, t, :]
            h = self.gru(self.action_stack(torch.cat([flatz, act_t], dim=-1)), h)
            embed = self.encoder(observations[:, t + 1])      # [BREAKOUT]
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, embed], dim=-1))
            )
            flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        return flatz, h

    @torch.no_grad()
    def act_step(self, prev_flatz, prev_h, prev_action, obs):
        # [BREAKOUT] obs: (B,4,10,10) -- durch Encoder statt direkt
        h = self.gru(self.action_stack(torch.cat([prev_flatz, prev_action], dim=-1)), prev_h)
        embed = self.encoder(obs)
        posterior_logits = self.logits_to_shape(
            self.posterior_model(torch.cat([h, embed], dim=-1))
        )
        flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        feat = torch.cat([flatz, h], dim=-1)
        return flatz, h, feat

    # imagine() bleibt IDENTISCH zu CartPole -- arbeitet rein im Latent-Raum.
    def imagine(self, start_flatz, start_h, actor, horizon=15):
        flatz, h = start_flatz, start_h
        feats, rewards, discounts, action_logits_list, actions = [], [], [], [], []
        for _ in range(horizon):
            feat = torch.cat([flatz, h], dim=-1)
            action_logits = actor(feat)
            a_idx = torch.distributions.Categorical(logits=action_logits).sample()
            a_onehot = F.one_hot(a_idx, num_classes=self.action_size).float()
            probs = F.softmax(action_logits, dim=-1)
            a = a_onehot + probs - probs.detach()
            h = self.gru(self.action_stack(torch.cat([flatz, a], dim=-1)), h)
            prior_logits = self.logits_to_shape(self.prior_model(h))
            flatz = self.flatten_latent(self.sample_straight_through(prior_logits))
            # [BREAKOUT] in der Imagination brauchen wir KEIN Bild zu dekodieren --
            # nur reward/continue. Wir rufen die pred_model-Heads direkt auf.
            pred_feat = self.pred_model(torch.cat([flatz, h], dim=-1))
            feats.append(torch.cat([flatz, h], dim=-1))
            rewards.append(self.reward_head(pred_feat))
            discounts.append(torch.sigmoid(self.continue_head(pred_feat)))
            action_logits_list.append(action_logits)
            actions.append(a)
        return {
            "features": torch.stack(feats, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "discounts": torch.stack(discounts, dim=1),
            "action_logits": torch.stack(action_logits_list, dim=1),
            "actions": torch.stack(actions, dim=1),
        }


# Actor und Critic sind IDENTISCH zu CartPole (operieren nur auf Latent-Features).
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_actions=ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x):
        return self.net(x)