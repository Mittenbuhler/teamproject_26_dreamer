"""
RSSM_cartpole.py — Sprint 4
Dreamer-v2 World Model mit Bild-Input (32x32 Graustufen).

Dreamer-Verluste:
  1. Reconstruction Loss  — Decoder(z,h) vs. echtes Frame
  2. Reward Loss          — Reward-Head(z,h) vs. echter Reward
  3. Continue Loss        — Continue-Head(z,h) vs. echter Continue-Flag
  4. KL Loss              — KL( q(z|h,o)  ||  p(z|h) )
                            Prior vs. Posterior (kategorisch, ST-Schätzer)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from Encoder import Encoder
from Decoder import Decoder
from CartPole import collect_episodes_image


# =============================================================================
# Hyperparameter
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESOLUTION   = 32       # Bildgröße
EMBED_DIM    = 32       # Encoder-Output-Größe (Linear 64→32)

C            = 8        # Anzahl kategorische Latentvariablen
K            = 8        # Klassen pro Variable
STOCH_SIZE   = C * K    # = 64

DETER_SIZE   = 256      # GRU Hidden-State h
HIDDEN_SIZE  = 256      # MLP-Zwischengröße

ACTION_SIZE  = 2        # CartPole: links / rechts (one-hot)

# Verlustgewichtungen
RECON_SCALE    = 1.0
REWARD_SCALE   = 1.0
CONTINUE_SCALE = 1.0
KL_SCALE       = 0.1
KL_BALANCE     = 0.8    # Dreamer-v2: 80% freie Prior, 20% freie Posterior

LR = 3e-4


# =============================================================================
# Sequence-Dataset (Bild-basiert)
# =============================================================================
class ImageSequenceDataset(torch.utils.data.Dataset):
    """
    Gibt Sequenzen mit Bildinput zurück.
    Shapes:
      frames:    (T+1, 1, H, W)   float32
      actions:   (T, 2)           float32
      rewards:   (T, 1)           float32
      continues: (T, 1)           float32
      full_states:(T+1, 4)        float32  (für Analyse)
    """

    def __init__(self, episodes: list[dict], seq_len: int = 16,
                 resolution: int = RESOLUTION):
        self.seqs = []
        self.resolution = resolution
        for ep in episodes:
            frames = np.stack(ep["frames"])                  # (T+1, res, res)
            frames = frames[:, np.newaxis, :, :]             # (T+1, 1, res, res)
            full   = np.stack(ep["full_states"])             # (T+1, 4)
            acts   = np.stack(ep["actions"])                 # (T, 2)
            rews   = np.stack(ep["rewards"])                 # (T, 1)
            conts  = np.stack(ep["continues"])               # (T, 1)
            T = len(acts)
            if T < 1:
                continue
            start = 0
            while start + seq_len <= T:
                self.seqs.append((
                    frames[start:start + seq_len + 1],
                    full  [start:start + seq_len + 1],
                    acts  [start:start + seq_len],
                    rews  [start:start + seq_len],
                    conts [start:start + seq_len],
                ))
                start += seq_len // 2   # 50%-Überlappung
            if not self.seqs:
                self.seqs.append((frames, full, acts, rews, conts))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        frames, full, acts, rews, conts = self.seqs[idx]
        return {
            "frames":      torch.from_numpy(frames.copy()).float(),
            "full_states": torch.from_numpy(full.copy()).float(),
            "actions":     torch.from_numpy(acts.copy()).float(),
            "rewards":     torch.from_numpy(rews.copy()).float(),
            "continues":   torch.from_numpy(conts.copy()).float(),
        }


# =============================================================================
# RSSM — Dreamer-v2 style, Bild-Input
# =============================================================================
class RSSM(nn.Module):
    """
    Recurrent State Space Model mit CNN-Encoder und CNN-Decoder.

    Zustandsrepräsentation:
      h  — deterministischer Zustand (GRU Hidden State)
      z  — stochastischer Zustand (kategorisch, C×K, straight-through)
      s  = [flatten(z), h]   — feature vector für alle Köpfe

    Prior:      p(z | h)         — sieht kein Bild
    Posterior:  q(z | h, e(o))   — sieht Encoder-Output e(o)
    """

    def __init__(self):
        super().__init__()

        # Encoder & Decoder — fixe Architektur (32x32 -> 32, 32 -> 32x32)
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim=STOCH_SIZE + DETER_SIZE)

        # Action preprocess: (z, a) → GRU-Input
        self.action_stack = nn.Sequential(
            nn.Linear(STOCH_SIZE + ACTION_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, DETER_SIZE),
        )

        # GRU
        self.gru = nn.GRUCell(DETER_SIZE, DETER_SIZE)

        # Prior p(z | h)
        self.prior_net = nn.Sequential(
            nn.Linear(DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, C * K),
        )

        # Posterior q(z | h, embed)
        self.posterior_net = nn.Sequential(
            nn.Linear(DETER_SIZE + EMBED_DIM, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, C * K),
        )

        # Reward Head
        self.reward_head = nn.Sequential(
            nn.Linear(STOCH_SIZE + DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        # Continue Head (logits für BCE)
        self.continue_head = nn.Sequential(
            nn.Linear(STOCH_SIZE + DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    # -----------------------------------------------------------------------
    # Hilfsmethoden
    # -----------------------------------------------------------------------
    def initial(self, batch_size: int, device: torch.device):
        z = torch.zeros(batch_size, STOCH_SIZE, device=device)
        h = torch.zeros(batch_size, DETER_SIZE, device=device)
        return z, h

    def _to_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, C*K) → (B, C, K)"""
        return logits.view(logits.shape[0], C, K)

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, K) → (B, C*K)"""
        return z.view(z.shape[0], C * K)

    def _straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-Through-Schätzer für kategorische Samples."""
        probs = F.softmax(logits, dim=-1)          # (B, C, K)
        B, _C, _K = probs.shape
        idx = torch.multinomial(probs.view(B * _C, _K), 1).squeeze(-1)
        onehot = F.one_hot(idx, num_classes=_K).float().view(B, _C, _K)
        # Forward: onehot, Backward: probs (Straight-Through)
        return onehot.detach() - probs.detach() + probs

    def _mode_onehot(self, logits: torch.Tensor) -> torch.Tensor:
        """Deterministische Auswahl (argmax) als one-hot."""
        idx = logits.argmax(dim=-1)                # (B, C)
        return F.one_hot(idx, num_classes=K).float()

    def _feature(self, flat_z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([flat_z, h], dim=-1)      # (B, STOCH+DETER)

    # -----------------------------------------------------------------------
    # Forward-Pass (Training)
    # -----------------------------------------------------------------------
    def observe_forward(self, frames: torch.Tensor, actions: torch.Tensor):
        """
        frames:  (B, T+1, 1, H, W)
        actions: (B, T, 2)

        Gibt alle nötigen Outputs für die Dreamer-Verluste zurück.
        """
        B, Tp1, C1, H, W = frames.shape
        T = actions.shape[1]

        # Alle Frames auf einmal durch Encoder (Effizienz)
        enc_all = self.encoder(frames.view(B * Tp1, C1, H, W))  # (B*(T+1), embed)
        enc_all = enc_all.view(B, Tp1, EMBED_DIM)               # (B, T+1, embed)

        flat_z, h = self.initial(B, frames.device)

        prior_logits_list     = []
        posterior_logits_list = []
        reconstructions       = []
        reward_preds          = []
        continue_preds        = []

        for t in range(T):
            a_t = actions[:, t, :]                              # (B, 2)

            # --- Recurrent Step ---
            inp = torch.cat([flat_z, a_t], dim=-1)             # (B, STOCH+2)
            inp = self.action_stack(inp)                        # (B, DETER)
            h   = self.gru(inp, h)                              # (B, DETER)

            # --- Prior p(z | h) ---
            prior_logits_flat = self.prior_net(h)               # (B, C*K)
            prior_logits      = self._to_categorical(prior_logits_flat)

            # --- Posterior q(z | h, e(o_{t+1})) ---
            e_next = enc_all[:, t + 1, :]                       # (B, embed)
            post_inp = torch.cat([h, e_next], dim=-1)
            post_logits_flat = self.posterior_net(post_inp)     # (B, C*K)
            post_logits      = self._to_categorical(post_logits_flat)

            # Sample posterior z (ST)
            z_post   = self._straight_through(post_logits)     # (B, C, K)
            flat_z   = self._flatten(z_post)                   # (B, C*K)

            # Feature für Predictions
            feat = self._feature(flat_z, h)                    # (B, STOCH+DETER)

            # --- Decoder (Reconstruction) ---
            recon = self.decoder(feat)                          # (B, 1, H, W)
            # --- Reward ---
            rew   = self.reward_head(feat)                      # (B, 1)
            # --- Continue ---
            cont  = self.continue_head(feat)                    # (B, 1)  [logit]

            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(post_logits)
            reconstructions.append(recon)
            reward_preds.append(rew)
            continue_preds.append(cont)

        return {
            "prior_logits":     torch.stack(prior_logits_list, dim=1),      # (B,T,C,K)
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),  # (B,T,C,K)
            "reconstructions":  torch.stack(reconstructions, dim=1),        # (B,T,1,H,W)
            "reward_preds":     torch.stack(reward_preds, dim=1),           # (B,T,1)
            "continue_preds":   torch.stack(continue_preds, dim=1),        # (B,T,1)
        }

    # -----------------------------------------------------------------------
    # Prior-Rollout (Imagination ohne echte Frames)
    # -----------------------------------------------------------------------
    def prior_rollout(self, frames: torch.Tensor, actions: torch.Tensor,
                      warmup_steps: int = 5):
        """
        Warmup: Posterior mit echten Frames.
        Rollout: nur Prior (keine echten Frames mehr).
        Gibt rekonstruierte Bilder des Imagination-Pfads zurück.
        """
        B, Tp1, C1, H, W = frames.shape
        T = actions.shape[1]
        flat_z, h = self.initial(B, frames.device)

        warmup_steps = min(warmup_steps, T)

        enc_all = self.encoder(frames.view(B * Tp1, C1, H, W))
        enc_all = enc_all.view(B, Tp1, EMBED_DIM)

        with torch.no_grad():
            # Warmup
            for t in range(warmup_steps):
                a_t = actions[:, t, :]
                inp = self.action_stack(torch.cat([flat_z, a_t], dim=-1))
                h   = self.gru(inp, h)
                e_next = enc_all[:, t + 1, :]
                post_logits = self._to_categorical(
                    self.posterior_net(torch.cat([h, e_next], dim=-1))
                )
                z_mode = self._mode_onehot(post_logits)
                flat_z = self._flatten(z_mode)

            # Prior-Rollout
            imagination_recons = []
            for t in range(warmup_steps, T):
                a_t = actions[:, t, :]
                inp = self.action_stack(torch.cat([flat_z, a_t], dim=-1))
                h   = self.gru(inp, h)
                prior_logits = self._to_categorical(self.prior_net(h))
                z_mode = self._mode_onehot(prior_logits)
                flat_z = self._flatten(z_mode)
                feat   = self._feature(flat_z, h)
                recon  = self.decoder(feat)
                imagination_recons.append(recon)

        if len(imagination_recons) == 0:
            return torch.zeros(B, 0, 1, H, W, device=frames.device)
        return torch.stack(imagination_recons, dim=1)   # (B, T-warmup, 1, H, W)


# =============================================================================
# Dreamer-Verluste
# =============================================================================
def compute_losses(out: dict, frames: torch.Tensor,
                   rewards: torch.Tensor, continues: torch.Tensor) -> tuple:
    """
    Dreamer-Verluste:
      L = recon_scale * L_recon
        + reward_scale * L_reward
        + continue_scale * L_continue
        + kl_scale * L_kl

    L_kl = kl_balance * KL(sg(q) || p) + (1-kl_balance) * KL(q || sg(p))
    (Dreamer-v2 KL-Balance: sg = stop_gradient)
    """
    # Zielbild: frames[:, 1:]  (echte nächste Frames)
    target_frames = frames[:, 1:, :, :, :]               # (B, T, 1, H, W)

    recons  = out["reconstructions"]                     # (B, T, 1, H, W)
    rew_p   = out["reward_preds"]                        # (B, T, 1)
    cont_p  = out["continue_preds"]                      # (B, T, 1)
    prior_l = out["prior_logits"]                        # (B, T, C, K)
    post_l  = out["posterior_logits"]                    # (B, T, C, K)

    # 1. Reconstruction Loss (MSE über Pixel)
    L_recon = F.mse_loss(recons, target_frames)

    # 2. Reward Loss (MSE)
    L_reward = F.mse_loss(rew_p, rewards)

    # 3. Continue Loss (BCE mit Logits)
    L_continue = F.binary_cross_entropy_with_logits(cont_p, continues)

    # 4. KL Loss mit Balance (Dreamer-v2)
    q_logprob = F.log_softmax(post_l, dim=-1)            # (B, T, C, K)
    p_logprob = F.log_softmax(prior_l, dim=-1)
    q_prob    = q_logprob.exp()

    # KL(q || p) — ganze KL
    kl_full = (q_prob * (q_logprob - p_logprob)).sum(-1)  # (B, T, C)

    # KL(sg(q) || p) — nur Prior lernt
    kl_prior_only = (q_prob.detach() * (q_logprob.detach() - p_logprob)).sum(-1)

    # KL(q || sg(p)) — nur Posterior lernt
    kl_post_only  = (q_prob * (q_logprob - p_logprob.detach())).sum(-1)

    kl_balanced = (KL_BALANCE * kl_prior_only
                   + (1 - KL_BALANCE) * kl_post_only).sum(-1).mean()  # sum C, mean B,T

    total = (RECON_SCALE    * L_recon
           + REWARD_SCALE   * L_reward
           + CONTINUE_SCALE * L_continue
           + KL_SCALE       * kl_balanced)

    kl_value = kl_full.sum(-1).mean().item()

    metrics = {
        "total_loss":      total.item(),
        "recon_loss":      L_recon.item(),
        "reward_loss":     L_reward.item(),
        "continue_loss":   L_continue.item(),
        "kl":              kl_value,
        "recon_max_err":   (recons - target_frames).abs().max().item(),
        "reward_mean_pred":rew_p.mean().item(),
        "continue_acc":    ((cont_p.sigmoid() > 0.5).float() == continues).float().mean().item(),
    }
    return total, metrics


# =============================================================================
# Training Loop
# =============================================================================
def train(
    n_iterations: int  = 15,
    n_seed_eps: int    = 10,
    n_new_eps: int     = 3,
    max_steps: int     = 100,
    seq_len: int       = 16,
    batch_size: int    = 8,
    resolution: int    = RESOLUTION,
):
    print(f"Device: {DEVICE}")
    print(f"Auflösung: {resolution}x{resolution}")
    print("=" * 70)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Replay Buffer
    replay_buffer = []
    print(f"Sammle {n_seed_eps} Startepisoden ...")
    replay_buffer += collect_episodes_image(env, n_seed_eps, max_steps, resolution, seed=42)

    model = RSSM().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell-Parameter: {n_params:,}")
    print("=" * 70)

    for iteration in range(n_iterations):
        # --- Neue Episoden sammeln ---
        new_eps = collect_episodes_image(env, n_new_eps, max_steps, resolution)
        replay_buffer += new_eps
        if len(replay_buffer) > 100:
            replay_buffer = replay_buffer[-100:]

        # --- Dataset & DataLoader ---
        dataset = ImageSequenceDataset(replay_buffer, seq_len=seq_len, resolution=resolution)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # --- Trainingsepoche ---
        agg, count = {}, 0
        model.train()
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            frames    = batch["frames"]       # (B, T+1, 1, H, W)
            actions   = batch["actions"]
            rewards   = batch["rewards"]
            continues = batch["continues"]

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out  = model.observe_forward(frames, actions)
                loss, metrics = compute_losses(out, frames, rewards, continues)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            scaler.step(opt)
            scaler.update()

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1

        if count == 0:
            continue
        avg = {k: v / count for k, v in agg.items()}

        print(
            f"Iter {iteration:02d} | Buffer={len(replay_buffer):3d} Eps | "
            f"L={avg['total_loss']:.4f}  recon={avg['recon_loss']:.4f}  "
            f"rew={avg['reward_loss']:.4f}  cont={avg['continue_loss']:.4f}  "
            f"kl={avg['kl']:.4f}  cont_acc={avg['continue_acc']:.3f}"
        )

    env.close()
    return model


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    model = train(n_iterations=15, n_seed_eps=10, n_new_eps=3)
    print("\nTraining abgeschlossen.")
    torch.save(model.state_dict(), "dreamer_sprint4.pt")
    print("Modell gespeichert: dreamer_sprint4.pt")
