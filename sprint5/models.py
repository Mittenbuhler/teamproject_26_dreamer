import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 4
K = 8
STOCHASTIC_SIZE = C * K
DETERMINISTIC_SIZE = 64   # war 8: zu klein, um die Aktionsdynamik zu tragen.
HIDDEN_SIZE = 256
OBSERVATION_SIZE = 4
ACTION_SIZE = 2


class RSSM(nn.Module):
    def __init__(
        self,
        categorical_size=C,
        class_size=K,
        stoch_size=STOCHASTIC_SIZE,
        deter_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
    ):
        super().__init__()
        self.C = categorical_size
        self.K = class_size
        self.stoch_size = stoch_size
        self.h_size = deter_size
        self.hid = hidden_size
        self.obs_size = obs_size
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
        self.observation_head = nn.Linear(self.hid, self.obs_size)
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

    def prediction_heads(self, flatz, h):
        feat = self.pred_model(torch.cat([flatz, h], dim=-1))
        return {
            "observation": self.observation_head(feat),
            "reward": self.reward_head(feat),
            "continuelogit": self.continue_head(feat),
        }

    def observe_forward(self, observations, actions):
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

            next_obs = observations[:, t + 1, :]
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, next_obs], dim=-1))
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
            next_obs = observations[:, t + 1, :]
            posterior_logits = self.logits_to_shape(
                self.posterior_model(torch.cat([h, next_obs], dim=-1))
            )
            flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        return flatz, h

    @torch.no_grad()
    def act_step(self, prev_flatz, prev_h, prev_action, obs):
        """Ein RSSM-Schritt beim Handeln in der echten Umgebung.

        Nimmt den vorigen Latent-Zustand (prev_flatz, prev_h), die zuletzt
        ausgefuehrte Aktion (one-hot) und die NEUE Beobachtung. Schreibt den
        deterministischen Zustand fort und korrigiert den stochastischen Zustand
        ueber den Posterior mit der neuen Beobachtung. Liefert den neuen
        (flatz, h) sowie das Policy-Feature [flatz, h].

        Damit sieht die Policy beim Handeln GENAU dieselbe Repraesentation wie
        im Imagination-Training -- keine separate ObsActor-Kodierung noetig.
        """
        h = self.gru(self.action_stack(torch.cat([prev_flatz, prev_action], dim=-1)), prev_h)
        posterior_logits = self.logits_to_shape(
            self.posterior_model(torch.cat([h, obs], dim=-1))
        )
        flatz = self.flatten_latent(self.mode_one_hot(posterior_logits))
        feat = torch.cat([flatz, h], dim=-1)
        return flatz, h, feat

    def imagine(self, start_flatz, start_h, actor, horizon=15):
        flatz, h = start_flatz, start_h
        feats, rewards, discounts, action_logits_list, actions = [], [], [], [], []

        for _ in range(horizon):
            feat = torch.cat([flatz, h], dim=-1)
            action_logits = actor(feat)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            a_idx = action_dist.sample()

            a_onehot = F.one_hot(a_idx, num_classes=self.action_size).float()
            probs = F.softmax(action_logits, dim=-1)
            a = a_onehot + probs - probs.detach()

            act_feat = self.action_stack(torch.cat([flatz, a], dim=-1))
            h = self.gru(act_feat, h)

            prior_logits = self.logits_to_shape(self.prior_model(h))
            prior_z = self.sample_straight_through(prior_logits)
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


class ObsActor(nn.Module):
    def __init__(self, obs_size, feat_size, policy=None, hidden_size=128, num_actions=ACTION_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feat_size),
            nn.ReLU(),
        )
        self.policy = policy if policy is not None else Actor(feat_size, hidden_size=hidden_size, num_actions=num_actions)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feat = self.encoder(x)
        return self.policy(feat)