import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SIZE = 2
PRIOR_SCALE = 0.5
KL_SCALE = 0.1
REWARD_SCALE = 2.0
CONTINUE_SCALE = 5.0
GAMMA = 0.995          # DreamerV2 Discount (Tab. D.1)
LAMBDA = 0.95          # lambda-target Parameter (Tab. D.1)
ALPHA_KL = 0.8
ACTOR_ENTROPY = 1e-3   # Actor entropy loss scale eta (Tab. D.1)
ACTOR_RHO = 1.0        # Actor gradient mixing: 1.0 = reines REINFORCE (Atari/diskret)
SLOW_CRITIC_UPDATE = 50   # Target-Netzwerk-Update-Intervall (haeufiger = stabilere Value-Targets)


def compute_world_model_loss(model_out, observations, rewards, continues,
                             priorscale=PRIOR_SCALE, klscale=KL_SCALE,
                             rewardscale=REWARD_SCALE, continuescale=CONTINUE_SCALE, alpha=ALPHA_KL,
                             mask=None):
    prior_logits = model_out["prior_logits"]
    posterior_logits = model_out["posterior_logits"]
    prior_preds = model_out["prior_predictions"]
    posterior_preds = model_out["posterior_predictions"]
    target_obs = observations[:, 1:, :]

    # Maskierte Loss-Helfer: mitteln nur ueber echte (nicht-gepaddete) Schritte.
    # Ohne Maske faellt alles auf das normale Mittel zurueck (Test-kompatibel).
    def masked_mse(pred, target):
        if mask is None:
            return F.mse_loss(pred, target)
        se = (pred - target) ** 2
        return (se * mask).sum() / (mask.sum() * pred.shape[-1] + 1e-8)

    def masked_bce(logit, target):
        if mask is None:
            return F.binary_cross_entropy_with_logits(logit, target)
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        return (bce * mask).sum() / (mask.sum() * logit.shape[-1] + 1e-8)

    obs_mse_post = masked_mse(posterior_preds["observation"], target_obs)
    rew_mse_post = masked_mse(posterior_preds["reward"], rewards)
    cont_loss_post = masked_bce(posterior_preds["continuelogit"], continues)

    obs_mse_prior = masked_mse(prior_preds["observation"], target_obs)
    rew_mse_prior = masked_mse(prior_preds["reward"], rewards)
    cont_loss_prior = masked_bce(prior_preds["continuelogit"], continues)

    posterior_loss = obs_mse_post + rewardscale * rew_mse_post + continuescale * cont_loss_post
    prior_loss = obs_mse_prior + rewardscale * rew_mse_prior + continuescale * cont_loss_prior

    def kl_divergence(logits_q, logits_p):
        q_log_probs = F.log_softmax(logits_q, dim=-1)
        p_log_probs = F.log_softmax(logits_p, dim=-1)
        q_probs = torch.exp(q_log_probs)
        return (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1).sum(dim=-1)

    if mask is None:
        kl_prior = kl_divergence(posterior_logits.detach(), prior_logits).mean()
        kl_post = kl_divergence(posterior_logits, prior_logits.detach()).mean()
    else:
        m = mask.squeeze(-1)  # (B, T)
        kl_prior = (kl_divergence(posterior_logits.detach(), prior_logits) * m).sum() / (m.sum() + 1e-8)
        kl_post = (kl_divergence(posterior_logits, prior_logits.detach()) * m).sum() / (m.sum() + 1e-8)
    kl = alpha * kl_prior + (1 - alpha) * kl_post

    total = posterior_loss + priorscale * prior_loss + klscale * kl
    metrics = {
        "total_loss": float(total.detach()),
        "kl": float(kl.detach()),
        "posterior_obs_mse": float(obs_mse_post.detach()),
        "prior_obs_mse": float(obs_mse_prior.detach()),
    }
    return total, metrics


def lambda_return(rewards, values, discounts, bootstrap, lam=LAMBDA):
    """lambda-Return nach DreamerV2 Gleichung 4.

    V^lambda_t = r_t + gamma_t * [ (1-lambda) * v(z_{t+1}) + lambda * V^lambda_{t+1} ]   fuer t < H
    V^lambda_H = v(z_H)

    Hier ist gamma_t bereits im 'discounts'-Tensor enthalten (= gamma * continue-prob),
    daher wird gamma nicht separat multipliziert. 'values' sind die Critic-Werte v(z_t)
    fuer t=0..H-1, 'bootstrap' ist v(z_H).
    """
    T = rewards.shape[1]
    # v(z_{t+1}) fuer t=0..T-1: die um eins verschobenen Werte, letzter = bootstrap.
    next_values = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)

    outs = []
    last = bootstrap  # V^lambda_H = v(z_H)
    for t in reversed(range(T)):
        last = rewards[:, t] + discounts[:, t] * ((1 - lam) * next_values[:, t] + lam * last)
        outs.append(last)
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
            loss, metrics = compute_world_model_loss(
                out, batch["observations"], batch["rewards"], batch["continues"],
                mask=batch.get("mask"),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return metrics


def train_actor_critic(world_model, actor, critic, actor_opt, critic_opt, replay_buffer,
                       imagination_horizon=15, warmup_steps=5, sample_episodes=4,
                       entropy_coeff=0.02, clip_norm=0.5, rho=1.0):
    if len(replay_buffer) == 0:
        return {}
    world_model.eval()
    actor.train()
    critic.train()

    if sample_episodes > 1 and hasattr(replay_buffer, "sample_episodes"):
        episodes = replay_buffer.sample_episodes(sample_episodes)
    else:
        episodes = [replay_buffer.sample_episode()]

    start_flatz, start_h = [], []
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
    rewards = imag["rewards"].squeeze(-1)         # (B, T)
    cont_prob = imag["discounts"].squeeze(-1)     # (B, T) = sigmoid(continuelogit)
    B, T, D = feats.shape

    # gamma_t = GAMMA * continue-Wahrscheinlichkeit (DreamerV2: discount = gamma * cont)
    discounts = GAMMA * cont_prob

    # Critic-Werte fuer alle imaginierten Zustaende.
    # Fuer den Critic-Loss werden die Features detached (der Critic braucht keinen
    # Gradienten durch das Weltmodell); der Actor-Pfad nutzt den vollen Graphen.
    flat_feats = feats.reshape(B * T, D)
    values = critic(flat_feats.detach()).view(B, T)

    # --- lambda-Return mit Target-Netzwerk (Gl. 4) ---
    # Target-Netzwerk (slow critic) fuer stabile Value-Targets.
    target_critic = getattr(train_actor_critic, "_target_critic", None)
    if target_critic is None:
        import copy
        target_critic = copy.deepcopy(critic)
        train_actor_critic._target_critic = target_critic
        train_actor_critic._update_count = 0

    with torch.no_grad():
        slow_values = target_critic(flat_feats).view(B, T)
        bootstrap = slow_values[:, -1]
        targets = lambda_return(rewards, slow_values, discounts, bootstrap)  # (B, T)

    # Discount-Gewichtung der Loss-Terme (kumulatives gamma_t, erster Faktor = 1).
    with torch.no_grad():
        discount_weight = torch.cumprod(
            torch.cat([torch.ones_like(discounts[:, :1]), discounts[:, :-1]], dim=1), dim=1
        )

    # --- CRITIC-LOSS (Gl. 5): MSE gegen sg(V^lambda), discount-gewichtet ---
    critic_loss = (discount_weight * (values - targets.detach()) ** 2).mean()

    # --- ACTOR-LOSS (Gl. 6): REINFORCE mit Baseline + Entropie ---
    actor_logits = imag["action_logits"]                     # (B, T, A)
    actions = imag["actions"]                                # (B, T, A) one-hot (ST)
    action_idx = actions.argmax(dim=-1)                      # (B, T)
    log_probs = F.log_softmax(actor_logits, dim=-1)
    selected_logp = log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Advantage = V^lambda - v(z_t) als Baseline (beide detached fuer REINFORCE-Pfad).
    advantage = (targets - values).detach()

    # Advantage-Normalisierung: haelt die REINFORCE-Update-Groesse konstant, egal
    # wie gross die Returns werden. Ohne das wachsen die Advantages mit steigendem
    # Reward (in CartPole bis ~400), die Updates werden immer aggressiver -- genau
    # dann, wenn die Policy schon gut ist -> Oszillation und Policy-Collapse.
    # Nur ueber die echten (discount-gewichteten relevanten) Schritte normalisieren.
    adv_mean = advantage.mean()
    adv_std = advantage.std() + 1e-8
    advantage = (advantage - adv_mean) / adv_std

    dist = torch.distributions.Categorical(logits=actor_logits)
    entropy = dist.entropy()                                 # (B, T)

    rho = ACTOR_RHO
    reinforce = -selected_logp * advantage                   # -ln p(a) * sg(V - v)
    dynamics = -targets                                       # -(V^lambda), Straight-Through-Pfad
    actor_term = rho * reinforce + (1.0 - rho) * dynamics - entropy_coeff * entropy

    # Discount-gewichtet mitteln (nur echte Vorhersageschritte zaehlen ueber gamma^t).
    actor_loss = (discount_weight * actor_term).mean()

    # --- Optimierung ---
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), clip_norm)
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
    critic_opt.step()

    # Target-Netzwerk periodisch aktualisieren (Gl. 5, alle 100 Schritte).
    train_actor_critic._update_count += 1
    if train_actor_critic._update_count % SLOW_CRITIC_UPDATE == 0:
        target_critic.load_state_dict(critic.state_dict())

    return {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "imagined_reward_mean": float(rewards.mean().detach()),
        "entropy": float(entropy.mean().detach()),
    }