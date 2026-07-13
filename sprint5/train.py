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


def compute_world_model_loss(model_out, observations, rewards, continues, priorscale=PRIOR_SCALE, klscale=KL_SCALE, rewardscale=REWARD_SCALE, continuescale=CONTINUE_SCALE):
    prior_logits = model_out["prior_logits"]
    posterior_logits = model_out["posterior_logits"]
    prior_preds = model_out["prior_predictions"]
    posterior_preds = model_out["posterior_predictions"]
    target_obs = observations[:, 1:, :]
    obs_mse_post = F.mse_loss(posterior_preds["observation"], target_obs)
    rew_mse_post = F.mse_loss(posterior_preds["reward"], rewards)
    cont_loss_post = F.binary_cross_entropy_with_logits(posterior_preds["continuelogit"], continues)
    obs_mse_prior = F.mse_loss(prior_preds["observation"], target_obs)
    rew_mse_prior = F.mse_loss(prior_preds["reward"], rewards)
    cont_loss_prior = F.binary_cross_entropy_with_logits(prior_preds["continuelogit"], continues)
    posterior_loss = obs_mse_post + rewardscale * rew_mse_post + continuescale * cont_loss_post
    prior_loss = obs_mse_prior + rewardscale * rew_mse_prior + continuescale * cont_loss_prior
    q_log_probs = F.log_softmax(posterior_logits, dim=-1)
    p_log_probs = F.log_softmax(prior_logits, dim=-1)
    q_probs = torch.exp(q_log_probs)
    kl = (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1).sum(dim=-1).mean()
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

    # Normalize return targets and advantages for more stable learning.
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

    # Backpropagate actor and critic losses together to avoid
    # "backward through the graph a second time" errors when parts
    # of the imagined trajectory share computation.
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