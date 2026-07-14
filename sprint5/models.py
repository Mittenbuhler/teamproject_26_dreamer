import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import Encoder
from .Decoder import Decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 4
K = 8
STOCHASTIC_SIZE = C * K
DETERMINISTIC_SIZE = 8
HIDDEN_SIZE = 64
ACTION_SIZE = 2

ENC_LATENT = 32  # Output-Dimension des Bild-Encoders (siehe encoder.py)


class RSSM(nn.Module):
    def __init__(
        self,
        categorical_size=C,
        class_size=K,
        stoch_size=STOCHASTIC_SIZE,
        deter_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_size=ENC_LATENT,
        action_size=ACTION_SIZE,
    ):
        super().__init__()
        self.C = categorical_size
        self.K = class_size
        self.stoch_size = stoch_size
        self.h_size = deter_size
        self.hid = hidden_size
        self.obs_size = obs_size  # = Encoder-Latentgröße, nicht die rohe Bildgröße
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
        self.posterior_model = nn.Sequential(
            nn.Linear(self.h_size + self.obs_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.C * self.K),
        )

        self.pred_model = nn.Sequential(
            nn.Linear(self.stoch_size + self.h_size, self.hid),
            nn.ReLU(),
        )

        # Bild-Encoder/-Decoder aus Sprint 4 (ersetzen den früheren
        # Linear-observation_head, der auf rohe 2D-Zustände ausgelegt war)
        self.image_encoder = Encoder()                     # (B,1,32,32) -> (B,32)
        self.image_decoder = Decoder(latent_dim=self.hid)  # (B,hid)     -> (B,1,32,32)

        self.reward_head = nn.Linear(self.hid, 1)
        self.continue_head = nn.Linear(self.hid, 1)

    def initial(self, batch_size=1, device=None):
        dev = device or DEVICE
        return (
            torch.zeros(batch_size, self.stoch_size, device=dev),
            torch.zeros(batch_size, self.h_size, device=dev),
        )

    def logits_to_shape(self, logits):
        B = logits.shape[0]
        return logits.view(B, self.C, self.K)

    def flatten_latent(self, z):
        B = z.shape[0]
        return z.view(B, self.C * self.K)

    def sample_straight_through(self, logits):
        probs = F.softmax(logits, dim=-1)
        B, C, K = probs.shape
        idx = torch.multinomial(probs.view(B * C, K), num_samples=1).squeeze(-1)
        onehot = F.one_hot(idx, num_classes=K).float().view(B, C, K)
        return onehot.detach() - probs.detach() + probs

    def mode_one_hot(self, logits):
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.K).float().to(logits.device)

    def encode_obs(self, obs_img):
        """Encodiert ein Bild-Batch (B,1,32,32) in den Latentraum (B, ENC_LATENT)."""
        return self.image_encoder(obs_img)

    def prediction_heads(self, flatz, h):
        feat = self.pred_model(torch.cat([flatz, h], dim=-1))
        return {
            "observation": self.image_decoder(feat),  # (B,1,32,32)
            "reward": self.reward_head(feat),
            "continuelogit": self.continue_head(feat),
        }

    def observe_forward(self, observations, actions):
        """
        observations: (B, T+1, 1, 32, 32) Bilder
        actions:      (B, T, action_size)
        """
        B, T = observations.shape[0], actions.shape[1]
        flatz, h = self.initial(batch_size=B, device=observations.device)

        prior_logits_list = []
        posterior_logits_list = []
        prior_preds = []
        posterior_preds = []

        for t in range(T):
            act_t = actions[:, t, :]
            h = self.gru(self.action_stack(torch.cat([flatz, act_t], dim=-1)), h)

            prior_logits = self.logits_to_shape(self.prior_model(h))
            prior_z = self.sample_straight_through(prior_logits)
            prior_flatz = self.flatten_latent(prior_z)
            prior_preds.append(self.prediction_heads(prior_flatz, h))

            next_obs_img = observations[:, t + 1]  # (B,1,32,32)
            encoded_obs = self.encode_obs(next_obs_img)  # (B, ENC_LATENT)
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, encoded_obs], dim=-1))
            )
            posterior_z = self.sample_straight_through(posterior_logits)
            posterior_flatz = self.flatten_latent(posterior_z)
            posterior_preds.append(self.prediction_heads(posterior_flatz, h))

            flatz = posterior_flatz
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

        def stack_dicts(dict_list):
            keys = dict_list[0].keys()
            return {k: torch.stack([d[k] for d in dict_list], dim=1) for k in keys}

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
            next_obs_img = observations[:, t + 1]
            encoded_obs = self.encode_obs(next_obs_img)
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, encoded_obs], dim=-1))
            )
            flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        return flatz, h

    @torch.no_grad()
    def step_online(self, flatz, h, action_onehot, next_obs_img):
        """Ein einzelner Online-RSSM-Schritt für die echte Environment-Interaktion:
        GRU-Update mit Aktion, danach Posterior-Update mit der neuen Beobachtung.
        Identisch zur Trainingslogik in observe_forward, nur fuer einen Zeitschritt."""
        act_feat = self.action_stack(torch.cat([flatz, action_onehot], dim=-1))
        h = self.gru(act_feat, h)
        encoded_obs = self.encode_obs(next_obs_img)
        posterior_logits = self.logits_to_shape(
            self.posterior_model(torch.cat([h, encoded_obs], dim=-1))
        )
        flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        return flatz, h

    def imagine(self, start_flatz, start_h, actor, horizon=15):
        flatz, h = start_flatz, start_h
        feats, rewards, discounts, action_logits_list, actions = [], [], [], [], []

        for _ in range(horizon):
            feat = torch.cat([flatz, h], dim=-1)
            action_logits = actor(feat)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            a_idx = action_dist.sample()
            a = F.one_hot(a_idx, num_classes=ACTION_SIZE).float()

            act_feat = self.action_stack(torch.cat([flatz, a], dim=-1))
            h = self.gru(act_feat, h)

            prior_logits = self.logits_to_shape(self.prior_model(h))
            prior_z = self.mode_one_hot(prior_logits)
            flatz = self.flatten_latent(prior_z)

            pred = self.prediction_heads(flatz, h)
            feats.append(torch.cat([flatz, h], dim=-1))
            rewards.append(pred["reward"])
            discounts.append(torch.sigmoid(pred["continuelogit"]))
            action_logits_list.append(action_logits)
            actions.append(a)

        return {
            "features": torch.stack(feats, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "discounts": torch.stack(discounts, dim=1),
            "action_logits": torch.stack(action_logits_list, dim=1),
            "actions": torch.stack(actions, dim=1),
        }


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_actions=ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)