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
ALPHA_KL = 0.8  # KL-Balancing Parameter alpha aus dem DreamerV2-Paper


def compute_world_model_loss(
    model_out, 
    observations, 
    rewards, 
    continues, 
    priorscale=PRIOR_SCALE, 
    klscale=KL_SCALE, 
    rewardscale=REWARD_SCALE, 
    continuescale=CONTINUE_SCALE,
    alpha=ALPHA_KL
):
    prior_logits = model_out["prior_logits"]
    posterior_logits = model_out["posterior_logits"]
    prior_preds = model_out["prior_predictions"]
    posterior_preds = model_out["posterior_predictions"]
    target_obs = observations[:, 1:, :]

    # Standard Vorhersageverluste (Posterior & Prior)
    obs_mse_post = F.mse_loss(posterior_preds["observation"], target_obs)
    rew_mse_post = F.mse_loss(posterior_preds["reward"], rewards)
    cont_loss_post = F.binary_cross_entropy_with_logits(posterior_preds["continuelogit"], continues)

    obs_mse_prior = F.mse_loss(prior_preds["observation"], target_obs)
    rew_mse_prior = F.mse_loss(prior_preds["reward"], rewards)
    cont_loss_prior = F.binary_cross_entropy_with_logits(prior_preds["continuelogit"], continues)

    posterior_loss = obs_mse_post + rewardscale * rew_mse_post + continuescale * cont_loss_post
    prior_loss = obs_mse_prior + rewardscale * rew_mse_prior + continuescale * cont_loss_prior

    # --- OPTIMIERUNG: KL-BALANCING (Algorithmus 2 aus dem Paper) ---
    def kl_divergence(logits_q, logits_p):
        q_log_probs = F.log_softmax(logits_q, dim=-1)
        p_log_probs = F.log_softmax(logits_p, dim=-1)
        q_probs = torch.exp(q_log_probs)
        # Summiere über Klassen (K) und Kategorien (C)
        return (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1).sum(dim=-1)

    # Trainiert den Prior in Richtung der Repräsentationen (detach posterior)
    kl_prior = kl_divergence(posterior_logits.detach(), prior_logits).mean()
    # Regularisiert Repräsentationen in Richtung des Priors (detach prior)
    kl_post = kl_divergence(posterior_logits, prior_logits.detach()).mean()

    kl = alpha * kl_prior + (1 - alpha) * kl_post
    # ----------------------------------------------------------------

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
    
    # --- OPTIMIERUNG: BEHEBUNG DES OFF-BY-ONE INDEX-FEHLERS ---
    # Wir verschieben die Werte, um v(z_{t+1}) sauber abzubilden.
    # Das letzte Element ist der Bootstrap-Wert der Critic.
    next_values = torch.cat([values[:, 1:], bootstrap.unsqueeze(-1)], dim=1)
    
    for t in reversed(range(T)):
        next_value = rewards[:, t] + gamma * discounts[:, t] * ((1 - lam) * next_values[:, t] + lam * next_value)
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
    rho=1.0,  # Rho-Mischparameter (1.0 = REINFORCE, 0.0 = Dynamics Backprop)
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

    # Generiert Trajektorien im latenten Raum des Modells
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

    # --- OPTIMIERUNG: KORREKTE CRITIC REGRESSION (Keine Normalisierung der Vorhersage) ---
    critic_loss = F.mse_loss(values, targets.detach())

    # Normalisierung der Advantages für stabile Actor-Updates
    advantages = (targets - values.detach())
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # ---------------------------------------------------------------------------------

    actor_logits = imag["action_logits"].reshape(B * T, ACTION_SIZE)
    actions = imag["actions"].reshape(B * T, ACTION_SIZE)
    action_idx = actions.argmax(dim=-1)

    log_probs = F.log_softmax(actor_logits, dim=-1)
    selected_logp = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1).view(B, T)

    entropy = torch.distributions.Categorical(logits=actor_logits).entropy().mean()

    # --- OPTIMIERUNG: VEREINTER ACTOR-LOSS (Gleichung 6 des Papers) ---
    # 1. REINFORCE Pfad (Nutzt normalisierte Advantages)
    reinforce_loss = -(selected_logp * advantages).mean()
    
    # 2. Dynamics Backpropagation Pfad (Maximiert Targets direkt über ST-Gradienten)
    targets_normalized = (targets - targets.mean()) / (targets.std() + 1e-8)
    dynamics_loss = -targets_normalized.mean()

    actor_loss = rho * reinforce_loss + (1.0 - rho) * dynamics_loss - entropy_coeff * entropy
    # ------------------------------------------------------------------

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