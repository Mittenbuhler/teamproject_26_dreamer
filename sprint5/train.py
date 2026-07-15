import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SIZE = 2
PRIOR_SCALE = 0.5
KL_SCALE = 0.1
REWARD_SCALE = 2.0
CONTINUE_SCALE = 1.0
GAMMA = 0.99
LAMBDA = 0.95

# --- KL-Balancing / Free-Bits (DreamerV2) ---
KL_BALANCE = 0.8   # Gewicht auf dem Term, der den Prior Richtung Posterior zieht
FREE_BITS = 1.0    # Nats/Zeitschritt, unterhalb derer der KL-Term nicht bestraft wird


def _categorical_kl(logits_a, logits_b):
    """KL(a || b) für kategoriale Verteilungen, letzte Achse = Klassen (K),
    danach über die C kategorialen Gruppen summiert.
    Input-Shape: (B, T, C, K) -> Output: (B, T)"""
    log_a = F.log_softmax(logits_a, dim=-1)
    log_b = F.log_softmax(logits_b, dim=-1)
    probs_a = torch.exp(log_a)
    kl = (probs_a * (log_a - log_b)).sum(dim=-1)  # (B, T, C)
    return kl.sum(dim=-1)  # (B, T)


def compute_world_model_loss(
    model_out,
    observations,
    rewards,
    continues,
    priorscale=PRIOR_SCALE,
    klscale=KL_SCALE,
    rewardscale=REWARD_SCALE,
    continuescale=CONTINUE_SCALE,
    kl_balance=KL_BALANCE,
    free_bits=FREE_BITS,
):
    prior_logits = model_out["prior_logits"]
    posterior_logits = model_out["posterior_logits"]
    prior_preds = model_out["prior_predictions"]
    posterior_preds = model_out["posterior_predictions"]
    target_obs = observations[:, 1:]

    obs_mse_post = F.mse_loss(posterior_preds["observation"], target_obs)
    rew_mse_post = F.mse_loss(posterior_preds["reward"], rewards)
    cont_loss_post = F.binary_cross_entropy_with_logits(posterior_preds["continuelogit"], continues)
    obs_mse_prior = F.mse_loss(prior_preds["observation"], target_obs)
    rew_mse_prior = F.mse_loss(prior_preds["reward"], rewards)
    cont_loss_prior = F.binary_cross_entropy_with_logits(prior_preds["continuelogit"], continues)
    posterior_loss = obs_mse_post + rewardscale * rew_mse_post + continuescale * cont_loss_post
    prior_loss = obs_mse_prior + rewardscale * rew_mse_prior + continuescale * cont_loss_prior

    # --- KL-Balancing ---
    # kl_post_to_prior: Posterior wird per stop-gradient fixiert, Gradient
    #   fliesst nur in den Prior -> zieht den Prior Richtung Posterior.
    # kl_prior_to_post: Prior wird per stop-gradient fixiert, Gradient
    #   fliesst nur in den Posterior -> haelt den Posterior in der Naehe des Priors.
    # kl_balance > 0.5 gewichtet die erste Richtung staerker, damit der Prior
    # schneller "aufholt" statt dass der Posterior zum Prior kollabiert.
    kl_post_to_prior = _categorical_kl(posterior_logits.detach(), prior_logits)
    kl_prior_to_post = _categorical_kl(posterior_logits, prior_logits.detach())

    # Free bits: KL erst bestrafen, wenn er ueber der Mindestschwelle liegt,
    # sonst bleibt der Gradient fuer diesen Anteil bei 0 -> verhindert Kollaps auf ~0.
    kl_post_to_prior = torch.clamp(kl_post_to_prior, min=free_bits)
    kl_prior_to_post = torch.clamp(kl_prior_to_post, min=free_bits)

    kl = kl_balance * kl_post_to_prior.mean() + (1 - kl_balance) * kl_prior_to_post.mean()

    total = posterior_loss + priorscale * prior_loss + klscale * kl
    metrics = {
        "total_loss": float(total.detach()),
        "kl": float(kl.detach()),
        "posterior_obs_mse": float(obs_mse_post.detach()),
        "prior_obs_mse": float(obs_mse_prior.detach()),
    }
    return total, metrics


def lambda_return(rewards, values, discounts, bootstrap, gamma=GAMMA, lam=LAMBDA):
    T = rewards.shape[1]
    next_value = bootstrap
    outs = []
    for t in reversed(range(T)):
        next_value = rewards[:, t] + gamma * discounts[:, t] * ((1 - lam) * values[:, t] + lam * next_value)
        outs.append(next_value)
    return torch.stack(list(reversed(outs)), dim=1)


def train_world_model(model, optimizer, replay_buffer, seq_len=16, batch_size=8, epochs_per_phase=1):
    from .data import SequenceDataset
    if len(replay_buffer) == 0:
        return {}
    dataset = SequenceDataset(list(replay_buffer.buffer), seq_len=seq_len)
    if len(dataset) == 0:
        return {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)
    metrics = {}
    model.train()
    for _ in range(epochs_per_phase):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model.observe_forward(batch["observations"], batch["actions"])
            loss, metrics = compute_world_model_loss(out, batch["observations"], batch["rewards"], batch["continues"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return metrics


def train_actor_critic(
    world_model,
    actor,
    critic,
    actor_opt,
    critic_opt,
    replay_buffer,
    imagination_horizon=15,
    warmup_steps=5,
    sample_episodes=4,
    entropy_coeff=0.02,
    clip_norm=0.5,
):
    if len(replay_buffer) == 0:
        return {}
    world_model.eval()
    actor.train()
    critic.train()

    if sample_episodes > 1 and hasattr(replay_buffer, "sample_episodes"):
        episodes = replay_buffer.sample_episodes(sample_episodes)
    else:
        episodes = [replay_buffer.sample_episode()]

    start_flatz = []
    start_h = []
    for ep in episodes:
        o = torch.tensor(np.array(ep["vis"], dtype=np.float32), device=DEVICE).unsqueeze(0)
        a = torch.tensor(np.array(ep["actions"], dtype=np.float32), device=DEVICE).unsqueeze(0)
        t0 = min(warmup_steps, a.shape[1] - 1)
        with torch.no_grad():
            sf, sh = world_model.posterior_start_state(o, a, t0=t0)
        start_flatz.append(sf)
        start_h.append(sh)

    start_flatz = torch.cat(start_flatz, dim=0)
    start_h = torch.cat(start_h, dim=0)

    imag = world_model.imagine(start_flatz, start_h, actor, horizon=imagination_horizon)

    feats = imag["features"]
    rewards = imag["rewards"].squeeze(-1)
    discounts = imag["discounts"].squeeze(-1)
    B, T, D = feats.shape

    flat_feats = feats.reshape(B * T, D)
    values = critic(flat_feats).view(B, T)

    with torch.no_grad():
        bootstrap = critic(feats[:, -1]).squeeze(-1)
        targets = lambda_return(rewards, values.detach(), discounts, bootstrap)

    targets = (targets - targets.mean()) / (targets.std() + 1e-8)
    values_normalized = (values - values.mean()) / (values.std() + 1e-8)

    critic_loss = F.mse_loss(values_normalized, targets)

    actor_logits = imag["action_logits"].reshape(B * T, ACTION_SIZE)
    actions = imag["actions"].reshape(B * T, ACTION_SIZE)
    action_idx = actions.argmax(dim=-1)

    log_probs = F.log_softmax(actor_logits, dim=-1)
    selected_logp = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1).view(B, T)
    advantages = (targets - values_normalized).detach()

    entropy = torch.distributions.Categorical(logits=actor_logits).entropy().mean()
    actor_loss = -(selected_logp * advantages).mean() - entropy_coeff * entropy

    actor_opt.zero_grad()
    critic_opt.zero_grad()
    total_loss = actor_loss + critic_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), clip_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
    actor_opt.step()
    critic_opt.step()

    return {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "imagined_reward_mean": float(rewards.mean().detach()),
    }