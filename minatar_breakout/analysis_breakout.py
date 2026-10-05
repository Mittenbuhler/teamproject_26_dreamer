"""
analysis_breakout.py — Analyse-Grafiken fuer das Breakout-Dreamer-Modell.

Erzeugt mehrere PNGs in einem plots/-Ordner (neben dieser Datei):
  1. wm_accuracy.png        Weltmodell-Genauigkeit (Pixel-Rekonstruktion, Reward, Continue)
  2. reconstruction.png     Echtes Frame vs. Posterior-Rekonstruktion, pro Kanal
  3. prior_rollout.png      Echtes Frame vs. getraeumtes (Prior) Frame ueber mehrere Schritte
  4. reward_histogram.png   Reward-Verteilung Actor vs. Zufall
  5. learning_curve.png     eval_return ueber Iterationen  (nur falls training_log.npy existiert)

Zuerst main ausfuehren: python -m minatar_breakout.main_breakout   (speichert *_breakout.pth)
Dann:                    python -m minatar_breakout.analysis_breakout
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .models_breakout import RSSM, Actor, Critic, DEVICE, OBS_CHANNELS, OBS_HW, ACTION_SIZE
from .data_breakout import collect_episodes, make_env, state_to_obs, onehot_action

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"                 # [BREAKOUT] eigener Plot-Ordner
CHANNEL_NAMES = ["paddle", "ball", "trail", "brick"]


def _load_models():
    wm = RSSM().to(DEVICE)
    feat = wm.stoch_size + wm.h_size
    actor = Actor(feat).to(DEVICE)
    wm_path = SCRIPT_DIR / "world_model_breakout.pth"
    actor_path = SCRIPT_DIR / "actor_breakout.pth"
    if not (wm_path.exists() and actor_path.exists()):
        raise FileNotFoundError(
            f"\n[FEHLER] Keine trainierten Gewichte in '{SCRIPT_DIR}' gefunden!\n"
            f"Bitte zuerst 'python -m minatar_breakout.main_breakout' ausfuehren."
        )
    wm.load_state_dict(torch.load(wm_path, map_location=DEVICE))
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    wm.eval(); actor.eval()
    print(f" -> Gewichte geladen aus {SCRIPT_DIR}")
    return wm, actor


# =============================================================================
# 1. Weltmodell-Genauigkeit
# =============================================================================
@torch.no_grad()
def plot_wm_accuracy(wm, episodes, save_path):
    import torch.nn.functional as F
    from .data_breakout import SequenceDataset
    ds = SequenceDataset(episodes, seq_len=16)
    if len(ds) == 0:
        print(" -> zu wenige/ kurze Episoden fuer wm_accuracy, ueberspringe.")
        return
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    post_pix, prior_pix, post_rew, prior_rew, post_cont, prior_cont, n = 0, 0, 0, 0, 0, 0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = wm.observe_forward(batch["observations"], batch["actions"])
        target = batch["observations"][:, 1:]
        # Pixel-Genauigkeit (Anteil korrekt vorhergesagter Bildzellen)
        pp = (torch.sigmoid(out["posterior_predictions"]["observation"]) > 0.5).float()
        qp = (torch.sigmoid(out["prior_predictions"]["observation"]) > 0.5).float()
        post_pix += (pp == target).float().mean().item()
        prior_pix += (qp == target).float().mean().item()
        post_rew += F.mse_loss(out["posterior_predictions"]["reward"], batch["rewards"]).item()
        prior_rew += F.mse_loss(out["prior_predictions"]["reward"], batch["rewards"]).item()
        pc = (out["posterior_predictions"]["continuelogit"].sigmoid() > 0.5).float()
        qc = (out["prior_predictions"]["continuelogit"].sigmoid() > 0.5).float()
        post_cont += (pc == batch["continues"]).float().mean().item()
        prior_cont += (qc == batch["continues"]).float().mean().item()
        n += 1

    post = [post_pix / n, post_cont / n]
    prior = [prior_pix / n, prior_cont / n]
    labels = ["Pixel-Genauigkeit", "Continue-Genauigkeit"]
    x = np.arange(len(labels)); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x - w/2, post, w, label="Posterior (mit Bild)", color="#2ecc71")
    ax1.bar(x + w/2, prior, w, label="Prior (traeumend)", color="#e67e22")
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="black", ls="--", lw=0.8)
    ax1.set_title("Genauigkeit (hoeher = besser)", fontweight="bold"); ax1.legend()

    ax2.bar(["Posterior", "Prior"], [post_rew/n, prior_rew/n], color=["#2ecc71", "#e67e22"])
    ax2.set_title("Reward-Vorhersagefehler (MSE, niedriger = besser)", fontweight="bold")

    plt.suptitle("Analyse 1: Weltmodell-Genauigkeit (Breakout)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f" -> gespeichert: {save_path}")


# =============================================================================
# 2. Bild-Rekonstruktion (echtes vs. rekonstruiertes Frame, pro Kanal)
# =============================================================================
@torch.no_grad()
def plot_reconstruction(wm, episodes, save_path, t_show=6):
    ep = max(episodes, key=lambda e: len(e["actions"]))
    vis = np.array(ep["vis"], dtype=np.float32)
    acts = np.array(ep["actions"], dtype=np.float32)
    o = torch.tensor(vis, device=DEVICE).unsqueeze(0)
    a = torch.tensor(acts, device=DEVICE).unsqueeze(0)
    out = wm.observe_forward(o, a)
    recon = torch.sigmoid(out["posterior_predictions"]["observation"][0]).cpu().numpy()  # (T,4,10,10)
    t = min(t_show, recon.shape[0] - 1)
    real_frame = vis[t + 1]      # (4,10,10)
    recon_frame = recon[t]       # (4,10,10)

    # 2 Zeilen (echt / rekonstruiert) x 4 Kanaele
    fig, axes = plt.subplots(2, OBS_CHANNELS, figsize=(2.3 * OBS_CHANNELS, 5))
    fig.suptitle(f"Analyse 2: Bild-Rekonstruktion pro Kanal (Schritt t={t+1})",
                 fontsize=13, fontweight="bold")
    for c in range(OBS_CHANNELS):
        axes[0, c].imshow(real_frame[c], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(CHANNEL_NAMES[c], fontsize=10)
        axes[1, c].imshow(recon_frame[c], cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("Echt", fontsize=11)
    axes[1, 0].set_ylabel("Rekonstruiert", fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f" -> gespeichert: {save_path}")


# =============================================================================
# 3. Prior-Rollout (freies Traeumen): echtes vs. getraeumtes Frame
# =============================================================================
@torch.no_grad()
def plot_prior_rollout(wm, episodes, save_path, warmup=5, n_steps=6, channel=1):
    """Zeigt den Kanal 'channel' (Default 1 = ball) ueber mehrere getraeumte Schritte."""
    ep = max(episodes, key=lambda e: len(e["actions"]))
    if len(ep["actions"]) < warmup + n_steps + 2:
        print(" -> Episode zu kurz fuer prior_rollout, ueberspringe.")
        return
    vis = np.array(ep["vis"], dtype=np.float32)
    acts = np.array(ep["actions"], dtype=np.float32)
    o = torch.tensor(vis, device=DEVICE).unsqueeze(0)
    a = torch.tensor(acts, device=DEVICE).unsqueeze(0)

    flatz, h = wm.posterior_start_state(o, a, t0=warmup)
    import torch.nn.functional as F
    imagined = []
    for i in range(n_steps):
        act_t = a[:, warmup + 1 + i, :]
        h = wm.gru(wm.action_stack(torch.cat([flatz, act_t], -1)), h)
        prior_logits = wm.logits_to_shape(wm.prior_model(h))
        flatz = wm.flatten_latent(wm.mode_one_hot(prior_logits))
        img = torch.sigmoid(wm.prediction_heads(flatz, h)["observation"][0]).cpu().numpy()
        imagined.append(img[channel])

    fig, axes = plt.subplots(2, n_steps, figsize=(2.2 * n_steps, 5))
    fig.suptitle(f"Analyse 3: Prior-Rollout, Kanal '{CHANNEL_NAMES[channel]}' (Traeumen ab t={warmup})",
                 fontsize=13, fontweight="bold")
    for i in range(n_steps):
        t = warmup + 2 + i
        real = vis[t][channel] if t < len(vis) else np.zeros((OBS_HW, OBS_HW))
        axes[0, i].imshow(real, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={t}", fontsize=9)
        axes[1, i].imshow(imagined[i], cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    axes[0, 0].set_ylabel("Echt", fontsize=11)
    axes[1, 0].set_ylabel("Getraeumt", fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f" -> gespeichert: {save_path}")


# =============================================================================
# 4. Reward-Histogramm: Actor vs. Zufall
# =============================================================================
@torch.no_grad()
def plot_reward_histogram(wm, actor, save_path, n_episodes=30, max_steps=500):
    def run(use_actor):
        rewards = []
        for i in range(n_episodes):
            env = make_env("breakout", seed=2000 + i)
            amap = env.minimal_action_set(); env.reset()
            flatz, h = wm.initial(batch_size=1, device=DEVICE)
            pa = torch.zeros(1, ACTION_SIZE, device=DEVICE)
            obs_t = torch.tensor(state_to_obs(env.state()), device=DEVICE).unsqueeze(0)
            flatz, h, feat = wm.act_step(flatz, h, pa, obs_t)
            total, done, steps = 0.0, False, 0
            while not done and steps < max_steps:
                if use_actor:
                    a_idx = torch.distributions.Categorical(logits=actor(feat)).sample().item()
                else:
                    a_idx = np.random.randint(ACTION_SIZE)
                r, done = env.act(amap[a_idx]); total += r; steps += 1
                pa = torch.tensor(onehot_action(a_idx), device=DEVICE).unsqueeze(0)
                obs_t = torch.tensor(state_to_obs(env.state()), device=DEVICE).unsqueeze(0)
                flatz, h, feat = wm.act_step(flatz, h, pa, obs_t)
            rewards.append(total)
        return np.array(rewards)

    trained = run(True); random_ = run(False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hi = max(trained.max(), random_.max(), 1)
    bins = np.arange(0, hi + 2, 1)
    ax.hist(random_, bins=bins, alpha=0.5, label=f"Zufall (Ø={random_.mean():.2f})", color="#95a5a6")
    ax.hist(trained, bins=bins, alpha=0.6, label=f"Actor (Ø={trained.mean():.2f})", color="#2ecc71")
    ax.set_xlabel("Reward pro Episode (zerstoerte Bricks)")
    ax.set_ylabel("Episoden-Anzahl")
    ax.set_title("Analyse 4: Reward-Verteilung Actor vs. Zufall", fontweight="bold")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f" -> gespeichert: {save_path}  (Actor Ø={trained.mean():.2f}, Zufall Ø={random_.mean():.2f})")


# =============================================================================
# 5. Lernkurve (nur falls Trainingslog vorhanden)
# =============================================================================
def plot_learning_curve(save_path):
    log_path = SCRIPT_DIR / "training_log_breakout.npy"
    if not log_path.exists():
        print(" -> kein training_log_breakout.npy gefunden, ueberspringe Lernkurve.")
        print("    (main_breakout speichert den Verlauf, falls aktiviert)")
        return
    returns = np.load(log_path)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(returns, color="#3498db", lw=1, alpha=0.5, label="eval_return")
    if len(returns) >= 5:
        kernel = np.ones(5) / 5
        smooth = np.convolve(returns, kernel, mode="valid")
        ax.plot(np.arange(len(smooth)) + 2, smooth, color="#e74c3c", lw=2, label="geglaettet (5)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("eval_return (Bricks)")
    ax.set_title("Analyse 5: Lernkurve", fontweight="bold"); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f" -> gespeichert: {save_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Breakout-Dreamer — Analyse-Grafiken")
    print("=" * 70)
    PLOT_DIR.mkdir(exist_ok=True)          # [BREAKOUT] plots/-Ordner anlegen
    print(f"Plots werden gespeichert in: {PLOT_DIR}")

    wm, actor = _load_models()

    print("\nSammle frische Analyse-Episoden ...")
    episodes = collect_episodes(None, actor, wm, n_episodes=30, max_steps=500, seed=999, epsilon=0.0)

    print("\n[1/5] Weltmodell-Genauigkeit ...")
    plot_wm_accuracy(wm, episodes, PLOT_DIR / "wm_accuracy.png")
    print("[2/5] Bild-Rekonstruktion ...")
    plot_reconstruction(wm, episodes, PLOT_DIR / "reconstruction.png")
    print("[3/5] Prior-Rollout ...")
    plot_prior_rollout(wm, episodes, PLOT_DIR / "prior_rollout.png")
    print("[4/5] Reward-Histogramm ...")
    plot_reward_histogram(wm, actor, PLOT_DIR / "reward_histogram.png")
    print("[5/5] Lernkurve ...")
    plot_learning_curve(PLOT_DIR / "learning_curve.png")

    print("\n" + "=" * 70)
    print(f"Fertig! Alle Grafiken in: {PLOT_DIR}")
    print("=" * 70)