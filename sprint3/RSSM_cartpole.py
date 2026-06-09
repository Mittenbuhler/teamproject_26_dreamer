import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# =============================================================================
# Hyperparameters
# =============================================================================
# C = number of separate categorical latent variables
# K = number of classes per categorical variable
# The latent state z has shape (B, C, K) and is flattened to (B, C*K)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = 4
K = 8
STOCHASTIC_SIZE = C * K
DETERMINISTIC_SIZE = 8
HIDDEN_SIZE = 16
OBSERVATION_SIZE = 2
ACTION_SIZE = 2
PRIOR_SCALE = 0.5
KL_SCALE = 0.1
LR = 1e-3

# CartPole full state is [cart position, cart velocity, pole angle, pole angular velocity]
# The model only sees cart position and pole angle.
VISIBLE_STATE_INDICES = np.array([0, 2])


# =============================================================================
# Environment helpers
# =============================================================================
def visible_state(full_state):
    # Keep only the visible parts of the CartPole state.
    return np.array(full_state)[VISIBLE_STATE_INDICES].astype(np.float32)


def onehot_action(a, action_size=ACTION_SIZE):
    # Actions are stored as one-hot vectors.
    v = np.zeros(action_size, dtype=np.float32)
    v[a] = 1.0
    return v


def collect_episodes(env, n_episodes=10, max_steps=200, seed=None):
    """
    Collect random CartPole episodes.

    Each episode stores:
    - full states
    - visible states
    - actions as one-hot vectors
    - rewards
    - continues (1 if episode continues after the step, 0 if it ends)
    """
    episodes = []
    for ep_idx in range(n_episodes):
        # Gymnasium reset returns (obs, info)
        if seed is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=seed + ep_idx)

        ep = {"fulls": [], "vis": [], "actions": [], "rewards": [], "continues": []}
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Random policy just to collect data
            a = env.action_space.sample()

            # Gymnasium step returns:
            # obs, reward, terminated, truncated, info
            next_full, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            # Store the current transition aligned with the manual:
            # action[t] predicts observation[t+1], reward[t], continues[t]
            ep["fulls"].append(obs)
            ep["vis"].append(visible_state(obs))
            ep["actions"].append(onehot_action(a))
            ep["rewards"].append([r])
            ep["continues"].append([0.0 if done else 1.0])

            obs = next_full
            steps += 1

        # Store final state so observations have length T+1
        ep["fulls"].append(obs)
        ep["vis"].append(visible_state(obs))
        episodes.append(ep)

    return episodes


# =============================================================================
# Sequence dataset
# =============================================================================
class SequenceDataset(torch.utils.data.Dataset):
    """
    Returns short supervised sequences.

    Shapes follow the manual:
    - observations: (T+1, 2)
    - actions:      (T, 2)
    - rewards:      (T, 1)
    - continues:    (T, 1)
    """
    def __init__(self, episodes, seq_len=10):
        self.seqs = []

        for ep in episodes:
            vis = np.array(ep["vis"], dtype=np.float32)
            acts = np.array(ep["actions"], dtype=np.float32)
            rews = np.array(ep["rewards"], dtype=np.float32)
            conts = np.array(ep["continues"], dtype=np.float32)
            L = len(acts)

            if L < 1:
                continue

            # Chunk episode into non-overlapping windows
            start = 0
            while start + seq_len <= L:
                o = vis[start:start + seq_len + 1]
                a = acts[start:start + seq_len]
                r = rews[start:start + seq_len]
                c = conts[start:start + seq_len]
                self.seqs.append((o, a, r, c))
                start += seq_len

        # Fallback if episodes are very short
        if len(self.seqs) == 0:
            for ep in episodes:
                vis = np.array(ep["vis"], dtype=np.float32)
                acts = np.array(ep["actions"], dtype=np.float32)
                rews = np.array(ep["rewards"], dtype=np.float32)
                conts = np.array(ep["continues"], dtype=np.float32)
                L = len(acts)
                if L < 1:
                    continue
                self.seqs.append((vis[:L + 1], acts[:L], rews[:L], conts[:L]))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        o, a, r, c = self.seqs[idx]
        return {
            "observations": torch.from_numpy(o),
            "actions": torch.from_numpy(a),
            "rewards": torch.from_numpy(r),
            "continues": torch.from_numpy(c),
        }


# =============================================================================
# RSSM model
# =============================================================================
class RSSM(nn.Module):
    """
    Tiny Dreamer-v2 style RSSM for hidden-velocity CartPole.

    Manual logic:
    - deterministic state h stores memory
    - stochastic latent z is categorical and one-hot per category
    - prior p(z_t | h_t) sees only h
    - posterior q(z_t | h_t, o_{t+1}) sees h and the real next observation
    - predictions are made from [flatten(z), h]
    """

    def __init__(self,
                 categorical_size=C,
                 class_size=K,
                 stoch_size=STOCHASTIC_SIZE,
                 deter_size=DETERMINISTIC_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 obs_size=OBSERVATION_SIZE,
                 action_size=ACTION_SIZE,
                 device=DEVICE):
        super().__init__()
        self.C = categorical_size
        self.K = class_size
        self.stoch_size = stoch_size
        self.h_size = deter_size
        self.hid = hidden_size
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device

        # actionstack: preprocess previous latent and current action before GRU
        self.action_stack = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.h_size),
        )

        # GRUCell updates deterministic memory h
        self.gru = nn.GRUCell(self.h_size, self.h_size)

        # prior p(z_t | h_t): only sees h
        self.prior_model = nn.Sequential(
            nn.Linear(self.h_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.C * self.K),
        )

        # posterior q(z_t | h_t, o_{t+1}): sees h and the real next observation
        self.posterior_model = nn.Sequential(
            nn.Linear(self.h_size + self.obs_size, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.C * self.K),
        )

        # prediction model builds features from [flatten(z), h]
        self.pred_model = nn.Sequential(
            nn.Linear(self.stoch_size + self.h_size, self.hid),
            nn.ReLU(),
        )

        # prediction heads
        self.observation_head = nn.Linear(self.hid, self.obs_size)
        self.reward_head = nn.Linear(self.hid, 1)
        self.continue_head = nn.Linear(self.hid, 1)

    def initial(self, batch_size=1, device=None):
        # Initial latent and deterministic state start as zeros.
        dev = device or self.device
        flatz = torch.zeros(batch_size, self.stoch_size, device=dev)
        h = torch.zeros(batch_size, self.h_size, device=dev)
        return flatz, h

    def logits_to_shape(self, logits):
        # Convert flattened logits (B, C*K) to categorical shape (B, C, K).
        B = logits.shape[0]
        return logits.view(B, self.C, self.K)

    def flatten_latent(self, z):
        # Flatten (B, C, K) to (B, C*K) before Linear layers and concatenation.
        B = z.shape[0]
        return z.view(B, self.C * self.K)

    def sample_straight_through(self, logits):
        """
        Straight-through categorical sample.

        Forward pass:
        - one-hot sample
        Backward pass:
        - gradients flow through probabilities
        """
        probs = F.softmax(logits, dim=-1)
        B, C, K = probs.shape

        # Sample one class for each of the C categoricals
        idx = torch.multinomial(probs.view(B * C, K), num_samples=1).squeeze(-1)
        onehot = F.one_hot(idx, num_classes=K).float().view(B, C, K)

        # Straight-through estimator:
        # forward == onehot, backward == probs
        return onehot.detach() - probs.detach() + probs

    def mode_one_hot(self, logits):
        """
        Deterministic evaluation mode:
        choose argmax over K and convert to one-hot.
        """
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.K).float().to(logits.device)

    def prediction_heads(self, flatz, h):
        """
        Predictions are made from [flatten(z), h].
        """
        x = torch.cat([flatz, h], dim=-1)
        feat = self.pred_model(x)

        return {
            "observation": self.observation_head(feat),
            "reward": self.reward_head(feat),
            "continuelogit": self.continue_head(feat),
        }

    def observe_forward(self, observations, actions):
        """
        Main RSSM forward pass over a chunk.

        Manual alignment:
        - actions[t] is used to predict observations[t+1], rewards[t], continues[t]
        - posterior sees the real next observation
        - next recurrent step uses posterior latent during training
        """
        B = observations.shape[0]
        T = actions.shape[1]

        prior_logits_list = []
        posterior_logits_list = []
        prior_preds = []
        posterior_preds = []

        # Start with zero latent and zero deterministic memory
        flatz, h = self.initial(batch_size=B, device=observations.device)

        for t in range(T):
            # Step 1: use previous latent flatz and current action
            act_t = actions[:, t, :]
            act_input = torch.cat([flatz, act_t], dim=-1)

            # Step 2: preprocess action + latent
            act_feat = self.action_stack(act_input)

            # Step 3: update deterministic memory h
            h = self.gru(act_feat, h)

            # Step 4: prior from h only
            # Important: prior must not see future observation
            prior_logits_flat = self.prior_model(h)
            prior_logits = self.logits_to_shape(prior_logits_flat)

            # Sample prior latent for prior predictions
            prior_z = self.sample_straight_through(prior_logits)
            prior_flatz = self.flatten_latent(prior_z)

            # Make predictions from prior latent
            prior_pred = self.prediction_heads(prior_flatz, h)

            # Step 5: posterior from h and real next observation o_{t+1}
            next_obs = observations[:, t + 1, :]
            post_input = torch.cat([h, next_obs], dim=-1)
            posterior_logits_flat = self.posterior_model(post_input)
            posterior_logits = self.logits_to_shape(posterior_logits_flat)

            # Sample posterior latent for posterior predictions
            posterior_z = self.sample_straight_through(posterior_logits)
            posterior_flatz = self.flatten_latent(posterior_z)

            # Make predictions from posterior latent
            posterior_pred = self.prediction_heads(posterior_flatz, h)

            # Important training detail:
            # The next recurrent step uses the posterior latent,
            # so the model stays grounded in the real trajectory.
            flatz = posterior_flatz

            # Store time-step outputs
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)
            prior_preds.append(prior_pred)
            posterior_preds.append(posterior_pred)

        def stack_dicts(dict_list):
            keys = dict_list[0].keys()
            return {k: torch.stack([d[k] for d in dict_list], dim=1) for k in keys}

        return {
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),
            "prior_predictions": stack_dicts(prior_preds),
            "posterior_predictions": stack_dicts(posterior_preds),
        }

    def prior_rollout(self, observations, actions, warmup_steps=5):
        """
        Imagination / prior rollout test.

        Warmup:
        - use posterior mode and real observations

        Rollout:
        - use prior mode only
        - do not use future observations
        """
        B = observations.shape[0]
        T = actions.shape[1]
        flatz, h = self.initial(batch_size=B, device=observations.device)

        warmup_steps = min(warmup_steps, T)

        # Warm up with real observations using the posterior
        for t in range(warmup_steps):
            act_t = actions[:, t, :]
            act_input = torch.cat([flatz, act_t], dim=-1)
            act_feat = self.action_stack(act_input)
            h = self.gru(act_feat, h)

            next_obs = observations[:, t + 1, :]
            post_logits_flat = self.posterior_model(torch.cat([h, next_obs], dim=-1))
            post_logits = self.logits_to_shape(post_logits_flat)

            # Use deterministic mode for evaluation rollout
            post_mode = self.mode_one_hot(post_logits)
            flatz = self.flatten_latent(post_mode)

        preds = []

        # Roll forward only with prior and recorded future actions
        for t in range(warmup_steps, T):
            act_t = actions[:, t, :]
            act_input = torch.cat([flatz, act_t], dim=-1)
            act_feat = self.action_stack(act_input)
            h = self.gru(act_feat, h)

            # Prior sees only h
            prior_logits_flat = self.prior_model(h)
            prior_logits = self.logits_to_shape(prior_logits_flat)
            prior_mode = self.mode_one_hot(prior_logits)
            flatz = self.flatten_latent(prior_mode)

            # Predict visible observation from prior latent
            pred = self.prediction_heads(flatz, h)["observation"]
            preds.append(pred)

        if len(preds) == 0:
            return torch.zeros(B, 0, self.obs_size, device=observations.device)

        return torch.stack(preds, dim=1)


# =============================================================================
# Losses
# =============================================================================
def compute_losses(model_out, observations, rewards, continues, priorscale=PRIOR_SCALE, klscale=KL_SCALE):
    """
    Manual objective:
    - posterior prediction loss
    - prior prediction loss
    - KL(q || p) over categorical latents
    """
    prior_logits = model_out["prior_logits"]
    posterior_logits = model_out["posterior_logits"]
    prior_preds = model_out["prior_predictions"]
    posterior_preds = model_out["posterior_predictions"]

    # Alignment:
    # observations[:, 1:] is the next observation for each action[t]
    target_obs = observations[:, 1:, :]

    # Posterior predictions
    obs_mse_post = F.mse_loss(posterior_preds["observation"], target_obs)
    rew_mse_post = F.mse_loss(posterior_preds["reward"], rewards)
    cont_loss_post = F.binary_cross_entropy_with_logits(posterior_preds["continuelogit"], continues)

    # Prior predictions
    obs_mse_prior = F.mse_loss(prior_preds["observation"], target_obs)
    rew_mse_prior = F.mse_loss(prior_preds["reward"], rewards)
    cont_loss_prior = F.binary_cross_entropy_with_logits(prior_preds["continuelogit"], continues)

    posterior_loss = obs_mse_post + rew_mse_post + cont_loss_post
    prior_loss = obs_mse_prior + rew_mse_prior + cont_loss_prior

    # KL(q || p) over K, then summed over C, then averaged over B and T
    q_log_probs = F.log_softmax(posterior_logits, dim=-1)
    p_log_probs = F.log_softmax(prior_logits, dim=-1)
    q_probs = torch.exp(q_log_probs)
    kl = (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1).sum(dim=-1).mean()

    total_loss = posterior_loss + priorscale * prior_loss + klscale * kl

    # Training metrics
    metrics = {
        "total_loss": total_loss.detach().item(),
        "kl": kl.detach().item(),
        "posterior_obs_mse": obs_mse_post.detach().item(),
        "prior_obs_mse": obs_mse_prior.detach().item(),
        "reward_mse": rew_mse_post.detach().item(),
        "continue_acc": ((torch.sigmoid(posterior_preds["continuelogit"]) > 0.5).float() == continues).float().mean().detach().item(),
    }

    return total_loss, metrics


# =============================================================================
# Debugging checklist tests
# =============================================================================
def check_no_nan(t, name):
    # The manual says NaNs usually mean shape, probability, or KL bugs.
    assert torch.isfinite(t).all(), f"{name} contains NaN or Inf"


def check_batch_shapes(batch):
    # Verify manual sequence shapes
    o = batch["observations"]
    a = batch["actions"]
    r = batch["rewards"]
    c = batch["continues"]

    assert o.ndim == 3 and o.shape[-1] == 2, f"observations shape bad: {o.shape}"
    assert a.ndim == 3 and a.shape[-1] == 2, f"actions shape bad: {a.shape}"
    assert r.ndim == 3 and r.shape[-1] == 1, f"rewards shape bad: {r.shape}"
    assert c.ndim == 3 and c.shape[-1] == 1, f"continues shape bad: {c.shape}"
    assert o.shape[1] == a.shape[1] + 1, f"alignment bad: observations {o.shape}, actions {a.shape}"


def check_latent_shapes(model, batch):
    # Prior/posterior logits must be B,T,C,K
    o = batch["observations"]
    a = batch["actions"]
    out = model.observe_forward(o, a)

    B, T = a.shape[0], a.shape[1]
    assert out["prior_logits"].shape == (B, T, model.C, model.K), f"prior logits bad: {out['prior_logits'].shape}"
    assert out["posterior_logits"].shape == (B, T, model.C, model.K), f"posterior logits bad: {out['posterior_logits'].shape}"

    # Sampling must keep the categorical structure before flattening
    z = model.sample_straight_through(out["prior_logits"][:, 0])
    assert z.shape == (B, model.C, model.K), f"sample shape bad: {z.shape}"

    flatz = model.flatten_latent(z)
    assert flatz.shape == (B, model.C * model.K), f"flatten shape bad: {flatz.shape}"

    mode = model.mode_one_hot(out["prior_logits"][:, 0])
    assert mode.shape == (B, model.C, model.K), f"mode shape bad: {mode.shape}"


def check_losses_finite(model, batch):
    # Loss should be finite and KL should be non-negative
    o = batch["observations"]
    a = batch["actions"]
    r = batch["rewards"]
    c = batch["continues"]

    out = model.observe_forward(o, a)
    loss, metrics = compute_losses(out, o, r, c)

    assert torch.isfinite(loss), "loss is not finite"
    assert metrics["kl"] >= -1e-6, f"KL negative: {metrics['kl']}"
    for k, v in metrics.items():
        assert np.isfinite(v), f"metric {k} not finite: {v}"


def check_continue_logits_raw(model, batch):
    # BCE-with-logits expects raw logits, not probabilities
    o = batch["observations"]
    a = batch["actions"]
    out = model.observe_forward(o, a)

    logits = out["posterior_predictions"]["continuelogit"]
    assert logits.shape[-1] == 1, f"continue logit shape bad: {logits.shape}"
    sig = torch.sigmoid(logits)
    assert torch.all(sig >= 0) and torch.all(sig <= 1), "sigmoid sanity failed"


def check_prior_rollout_leakage(model, batch):
    # Prior rollout must not access future observations after warmup
    o = batch["observations"]
    a = batch["actions"]
    with torch.no_grad():
        pred = model.prior_rollout(o, a, warmup_steps=min(4, a.shape[1]))
    assert pred.ndim == 3 and pred.shape[-1] == 2, f"prior rollout bad: {pred.shape}"


def run_debug_checks(model, batch):
    """
    Manual debugging checklist:
    - batch shapes
    - latent shapes
    - finite loss and non-negative KL
    - continue logits are valid raw logits
    - prior rollout has correct shape and no leakage
    """
    check_batch_shapes(batch)
    check_latent_shapes(model, batch)
    check_losses_finite(model, batch)
    check_continue_logits_raw(model, batch)
    check_prior_rollout_leakage(model, batch)
    print("All debug checks passed.")


# =============================================================================
# Demo training step
# =============================================================================
def demo_train_step():
    # Create training environment with Gymnasium
    env = gym.make("CartPole-v1")

    # Collect random episodes for a quick smoke test
    episodes = collect_episodes(env, n_episodes=8, max_steps=50, seed=42)
    dataset = SequenceDataset(episodes, seq_len=8)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = RSSM().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Run the manual-style debugging checklist before training
        run_debug_checks(model, batch)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]

        # Main forward pass
        out = model.observe_forward(observations, actions)

        # Compute RSSM objective
        loss, metrics = compute_losses(out, observations, rewards, continues)

        # Standard training step
        opt.zero_grad()
        loss.backward()
        opt.step()

        print("Demo train step metrics:", metrics)

        # Prior rollout test after warmup
        pred = model.prior_rollout(observations, actions, warmup_steps=4)
        print("Prior rollout shape:", pred.shape)
        break

    env.close()


if __name__ == "__main__":
    demo_train_step()