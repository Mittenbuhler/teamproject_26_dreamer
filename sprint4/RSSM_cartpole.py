"""
RSSM_cartpole.py — Sprint 4 (optimiert)
Dreamer-v2 World Model mit Bild-Input (32x32 Graustufen).
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

RESOLUTION   = 32
EMBED_DIM    = 32       # Encoder-Output (Linear 64→32)

C            = 8        # Kategorische Latentvariablen
K            = 8        # Klassen pro Variable
STOCH_SIZE   = C * K    # 64

DETER_SIZE   = 256      # GRU Hidden-State
HIDDEN_SIZE  = 256

ACTION_SIZE  = 2

RECON_SCALE    = 1.0
REWARD_SCALE   = 1.0
CONTINUE_SCALE = 1.0
KL_SCALE       = 0.1
KL_BALANCE     = 0.8    # Dreamer-v2: 80% freie Prior, 20% freie Posterior
KL_FREE_BITS   = 0.5   # Min-KL pro Kategorie — verhindert Posterior-Kollaps

LR         = 3e-4
GRAD_CLIP  = 10.0
GRAD_STEPS = 40         # Gradient-Updates pro Iteration
BATCH_SIZE = 32         # Stabile Gradienten


# =============================================================================
# Dataset & Sampling
# =============================================================================
class ImageSequenceDataset(torch.utils.data.Dataset):
    """
    Gibt überlappende Sequenzen aus dem Replay Buffer zurück.
    Shapes pro Sample:
      frames:     (T+1, 1, H, W)  float32  [0,1]
      full_states:(T+1, 4)        float32  [pos, vel, angle, angvel]
      actions:    (T, 2)          float32  one-hot
      rewards:    (T, 1)          float32
      continues:  (T, 1)          float32  1=weiter, 0=Ende
    """
    def __init__(self, episodes: list[dict], seq_len: int = 16):
        self.seqs = []
        for ep in episodes:
            frames = np.stack(ep["frames"])[:, np.newaxis, :, :]
            full   = np.stack(ep["full_states"])
            acts   = np.stack(ep["actions"])
            rews   = np.stack(ep["rewards"])
            conts  = np.stack(ep["continues"])
            T = len(acts)
            if T < 1:
                continue
            # Episoden kürzer als seq_len werden übersprungen — alle Sequenzen
            # im Dataset haben exakt (seq_len+1) Frames, damit sample_batch
            # problemlos torch.stack aufrufen kann.
            start = 0
            while start + seq_len <= T:
                self.seqs.append((
                    frames[start:start + seq_len + 1],
                    full  [start:start + seq_len + 1],
                    acts  [start:start + seq_len],
                    rews  [start:start + seq_len],
                    conts [start:start + seq_len],
                ))
                start += seq_len // 2   # 50% Überlappung

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


def sample_batch(dataset: ImageSequenceDataset, batch_size: int,
                 device: torch.device) -> dict:
    """Zieht einen zufälligen Batch ohne DataLoader-Overhead."""
    idx   = torch.randint(len(dataset), (batch_size,)).tolist()
    batch = [dataset[i] for i in idx]
    return {k: torch.stack([b[k] for b in batch]).to(device) for k in batch[0]}


# =============================================================================
# RSSM
# =============================================================================
class RSSM(nn.Module):
    """
    Recurrent State Space Model (Dreamer-v2).

    Zustandsrepräsentation:
      h  — deterministischer Zustand (GRU Hidden State)
      z  — stochastischer Zustand (kategorisch, C×K, straight-through)
      feat = [flatten(z), h]  — Input für alle Decoder-Köpfe

    Prior:     p(z | h)        — ohne Bild
    Posterior: q(z | h, e(o)) — mit Encoder-Output
    """
    def __init__(self):
        super().__init__()

        self.encoder    = Encoder()
        self.decoder    = Decoder(latent_dim=32)
        # Projiziert feat=[z,h] (STOCH+DETER=320) auf 32 dim vor dem Decoder
        self.feat_proj  = nn.Linear(STOCH_SIZE + DETER_SIZE, 32)
        self.embed_norm = nn.LayerNorm(EMBED_DIM)

        self.action_stack = nn.Sequential(
            nn.Linear(STOCH_SIZE + ACTION_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, DETER_SIZE),
        )
        self.gru = nn.GRUCell(DETER_SIZE, DETER_SIZE)

        self.prior_net = nn.Sequential(
            nn.Linear(DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, C * K),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(DETER_SIZE + EMBED_DIM, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, C * K),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(STOCH_SIZE + DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(STOCH_SIZE + DETER_SIZE, HIDDEN_SIZE),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    # -----------------------------------------------------------------------
    def initial(self, batch_size: int, device: torch.device):
        return (torch.zeros(batch_size, STOCH_SIZE, device=device),
                torch.zeros(batch_size, DETER_SIZE, device=device))

    def _to_categorical(self, logits):
        return logits.view(logits.shape[0], C, K)

    def _flatten(self, z):
        return z.view(z.shape[0], C * K)

    def _straight_through(self, logits):
        """Kategorisches Sample mit Straight-Through-Gradienten."""
        probs  = F.softmax(logits, dim=-1)           # (B, C, K)
        B, _C, _K = probs.shape
        idx    = torch.multinomial(probs.view(B * _C, _K), 1).squeeze(-1)
        onehot = F.one_hot(idx, _K).float().view(B, _C, _K)
        return onehot.detach() - probs.detach() + probs  # ST-Trick

    def _mode_onehot(self, logits):
        """Deterministisches argmax als one-hot (für Evaluation/Rollout)."""
        return F.one_hot(logits.argmax(dim=-1), K).float()

    def _feature(self, flat_z, h):
        return torch.cat([flat_z, h], dim=-1)         # (B, STOCH+DETER)

    # -----------------------------------------------------------------------
    def observe_forward(self, frames: torch.Tensor, actions: torch.Tensor):
        """
        Kompletter Forward-Pass über eine Sequenz.
        frames:  (B, T+1, 1, H, W)
        actions: (B, T, 2)
        """
        B, Tp1, C1, H, W = frames.shape
        T = actions.shape[1]

        # Encoder auf alle Frames gleichzeitig (effizienter als Schleife)
        enc_all = self.encoder(frames.view(B * Tp1, C1, H, W))
        enc_all = self.embed_norm(enc_all).view(B, Tp1, EMBED_DIM)

        flat_z, h = self.initial(B, frames.device)

        prior_logits_list, posterior_logits_list = [], []
        reconstructions, reward_preds, continue_preds = [], [], []

        for t in range(T):
            # Recurrent Step
            inp  = self.action_stack(torch.cat([flat_z, actions[:, t]], dim=-1))
            h    = self.gru(inp, h)

            # Prior p(z | h)
            prior_logits = self._to_categorical(self.prior_net(h))

            # Posterior q(z | h, e(o_{t+1}))
            e_next      = enc_all[:, t + 1]
            post_logits = self._to_categorical(
                self.posterior_net(torch.cat([h, e_next], dim=-1)))

            # Sample z aus Posterior (Straight-Through)
            z_post = self._straight_through(post_logits)
            flat_z = self._flatten(z_post)
            feat   = self._feature(flat_z, h)

            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(post_logits)
            reconstructions.append(self.decoder(self.feat_proj(feat)))
            reward_preds.append(self.reward_head(feat))
            continue_preds.append(self.continue_head(feat))

        return {
            "prior_logits":     torch.stack(prior_logits_list,     dim=1),  # (B,T,C,K)
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),  # (B,T,C,K)
            "reconstructions":  torch.stack(reconstructions,        dim=1),  # (B,T,1,H,W)
            "reward_preds":     torch.stack(reward_preds,           dim=1),  # (B,T,1)
            "continue_preds":   torch.stack(continue_preds,         dim=1),  # (B,T,1)
        }

    # -----------------------------------------------------------------------
    def prior_rollout(self, frames: torch.Tensor, actions: torch.Tensor,
                      warmup_steps: int = 5):
        """
        Imagination: Warmup mit Posterior, dann freier Prior-Rollout.
        Gibt rekonstruierte Frames des imaginierten Pfads zurück.
        """
        B, Tp1, C1, H, W = frames.shape
        T = actions.shape[1]
        warmup_steps = min(warmup_steps, T)

        enc_all = self.embed_norm(
            self.encoder(frames.view(B * Tp1, C1, H, W))
        ).view(B, Tp1, EMBED_DIM)

        flat_z, h = self.initial(B, frames.device)

        with torch.no_grad():
            # Warmup — Posterior, sieht echte Frames
            for t in range(warmup_steps):
                inp = self.action_stack(torch.cat([flat_z, actions[:, t]], dim=-1))
                h   = self.gru(inp, h)
                post_logits = self._to_categorical(
                    self.posterior_net(torch.cat([h, enc_all[:, t + 1]], dim=-1)))
                flat_z = self._flatten(self._mode_onehot(post_logits))

            # Rollout — nur Prior, keine echten Frames mehr ("Träumen")
            imagination = []
            for t in range(warmup_steps, T):
                inp = self.action_stack(torch.cat([flat_z, actions[:, t]], dim=-1))
                h   = self.gru(inp, h)
                prior_logits = self._to_categorical(self.prior_net(h))
                flat_z = self._flatten(self._mode_onehot(prior_logits))
                imagination.append(self.decoder(self.feat_proj(self._feature(flat_z, h))))

        if not imagination:
            return torch.zeros(B, 0, 1, H, W, device=frames.device)
        return torch.stack(imagination, dim=1)   # (B, T-warmup, 1, H, W)


# =============================================================================
# Verluste
# =============================================================================
def compute_losses(out: dict, frames: torch.Tensor,
                   rewards: torch.Tensor, continues: torch.Tensor) -> tuple:
    """
    Dreamer-Verluste (spec-konform):
      L = L_recon + L_reward + L_continue + kl_scale * L_KL

    Reconstruction Loss: MSE auf invertierten Frames.

    Warum invertieren (1 - frame) vor dem MSE?
      CartPole-Hintergrund ist zu 91% weiß (Pixelwert ≈ 1.0).
      Der MSE-Gradient auf einem weißen Pixel ist ~17x kleiner als auf einem
      dunklen Pixel wenn der Decoder falsch liegt — weil (pred - 1.0)² für
      pred=0.95 nur 0.0025 ergibt, aber (pred - 0.1)² für pred=0.95 schon
      0.7225. Das führt dazu dass der Decoder "alles weiß" lernt und den Stab
      völlig ignoriert.
      Nach Invertierung: Hintergrund=0, Stab/Wagen=hell. MSE behandelt alle
      Pixel fair. Der Loss bleibt strukturell identisch zur Spec
      ("decodiertes Bild vs. echtes Bild") — nur das Farbschema ist gedreht.
    """
    # Frames invertieren: Stab/Wagen hell, Hintergrund schwarz
    target = 1.0 - frames[:, 1:, :, :, :]          # (B, T, 1, H, W) — echtes nächstes Frame
    recons = 1.0 - out["reconstructions"]            # (B, T, 1, H, W) — Decoder-Output

    # 1. Reconstruction Loss: MSE (spec-konform, auf invertierten Frames)
    L_recon = F.mse_loss(recons, target)

    # 2. Reward Loss: MSE
    L_reward = F.mse_loss(out["reward_preds"], rewards)

    # 3. Continue Loss: BCE mit Logits
    L_continue = F.binary_cross_entropy_with_logits(out["continue_preds"], continues)

    # 4. KL Loss: Dreamer-v2 Balance + Free Bits
    #    q = Posterior q(z|h,o),  p = Prior p(z|h)
    #    Balance: 0.8 * KL(sg(q)||p) + 0.2 * KL(q||sg(p))
    #    -> Prior lernt 80% des KL-Signals, Posterior 20%
    #    Free Bits: KL unter 0.5 nats/Kategorie trägt keinen Gradienten bei
    #    -> verhindert triviales Kollabieren des Posteriors auf den Prior
    q_lp = F.log_softmax(out["posterior_logits"], dim=-1)  # (B, T, C, K)
    p_lp = F.log_softmax(out["prior_logits"],     dim=-1)
    q_p  = q_lp.exp()

    kl_full       = (q_p * (q_lp - p_lp)).sum(-1)                          # (B, T, C)
    kl_prior_only = (q_p.detach() * (q_lp.detach() - p_lp)).sum(-1)        # nur Prior lernt
    kl_post_only  = (q_p * (q_lp - p_lp.detach())).sum(-1)                 # nur Posterior lernt

    kl_balanced = (
        KL_BALANCE       * kl_prior_only.clamp(min=KL_FREE_BITS)
        + (1-KL_BALANCE) * kl_post_only .clamp(min=KL_FREE_BITS)
    ).sum(-1).mean()   # sum über C Kategorien, mean über B*T

    total = (RECON_SCALE    * L_recon
           + REWARD_SCALE   * L_reward
           + CONTINUE_SCALE * L_continue
           + KL_SCALE       * kl_balanced)

    metrics = {
        "total_loss":    total.item(),
        "recon_loss":    L_recon.item(),
        "reward_loss":   L_reward.item(),
        "continue_loss": L_continue.item(),
        "kl":            kl_full.sum(-1).mean().item(),
        "continue_acc":  ((out["continue_preds"].sigmoid() > 0.5).float()
                          == continues).float().mean().item(),
    }
    return total, metrics


# =============================================================================
# Training
# =============================================================================
def train(
    n_iterations: int = 250,    # Runter setzten für schnelleres Training
    n_seed_eps: int   = 50,    
    n_new_eps: int    = 5,
    max_steps: int    = 200,  
    seq_len: int      = 16,
    batch_size: int   = BATCH_SIZE,
    grad_steps: int   = GRAD_STEPS,
    resolution: int   = RESOLUTION,
):
    print(f"Device: {DEVICE}")
    print(f"Auflösung: {resolution}x{resolution}")
    print(f"Loss: MSE auf invertierten Frames | Batch={batch_size} | Grad-Steps={grad_steps} | Clip={GRAD_CLIP}")
    print("=" * 70)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    replay_buffer = []
    print(f"Sammle {n_seed_eps} Startepisoden (max {max_steps} Schritte) ...")
    replay_buffer += collect_episodes_image(env, n_seed_eps, max_steps, resolution, seed=42)
    total_steps = sum(len(ep["actions"]) for ep in replay_buffer)
    print(f"  → {len(replay_buffer)} Episoden, {total_steps} Schritte gesamt")

    model  = RSSM().to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=LR, eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_params:,}")
    print("=" * 70)

    for iteration in range(n_iterations):
        # Neue Episoden sammeln und Buffer begrenzen
        replay_buffer = (replay_buffer
                         + collect_episodes_image(env, n_new_eps, max_steps, resolution))[-200:]

        dataset = ImageSequenceDataset(replay_buffer, seq_len=seq_len)

        # Mehrere unabhängige Gradient-Updates pro Iteration
        agg, count = {}, 0
        model.train()
        for _ in range(grad_steps):
            batch = sample_batch(dataset, batch_size, DEVICE)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out          = model.observe_forward(batch["frames"], batch["actions"])
                loss, metrics = compute_losses(
                    out, batch["frames"], batch["rewards"], batch["continues"])

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1

        avg = {k: v / count for k, v in agg.items()}
        print(
            f"Iter {iteration:02d} | Buffer={len(replay_buffer):3d} | "
            f"L={avg['total_loss']:.4f}  recon={avg['recon_loss']:.4f}  "
            f"rew={avg['reward_loss']:.4f}  cont={avg['continue_loss']:.4f}  "
            f"kl={avg['kl']:.4f}  cont_acc={avg['continue_acc']:.3f}"
        )

    env.close()
    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "dreamer_sprint4.pt")
    print("\nModell gespeichert: dreamer_sprint4.pt")