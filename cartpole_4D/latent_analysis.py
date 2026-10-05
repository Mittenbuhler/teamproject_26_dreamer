"""
latent_analysis.py — Sprint 5 (auf Vektor-Modell portiert)
Latent-Space-Analyse des trainierten Dreamer-Modells.

Idee: ein separates PyTorch-NN "sondiert" den RSSM-Latentraum (stoch + deter)
und rekonstruiert daraus die 4 echten CartPole-Zustandsvariablen. Das Modell
selbst sieht diese Werte nie in der Analyse -- sie werden aus ep["fulls"]
entnommen. (Da das Weltmodell jetzt selbst 4D sieht, ist "fulls" == "vis".)

Ausführen mit: python3 -m sprint5.latent_analysis
Voraussetzung: python -m sprint5.main wurde ausgeführt (world_model.pth /
actor.pth im sprint5/-Ordner).
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from .models import RSSM, Actor, Critic, DEVICE
from .data import collect_episodes
from .analysis_state_model import _open_loop_rollout_fixed_actions

SCRIPT_DIR = Path(__file__).resolve().parent

# Reihenfolge entspricht Gymnasium CartPole: [x, xdot, theta, thetadot]
STATE_NAMES = ["Position (x)", "Geschw. (ẋ)", "Winkel (θ)", "Winkelgesch. (θ̇)"]


# =============================================================================
# 1. SETUP
# =============================================================================
def load_or_train_model():
    world_model = RSSM().to(DEVICE)
    feat_size = world_model.stoch_size + world_model.h_size
    actor = Actor(feat_size).to(DEVICE)

    wm_path = SCRIPT_DIR / "world_model.pth"
    actor_path = SCRIPT_DIR / "actor.pth"

    if not (wm_path.exists() and actor_path.exists()):
        raise FileNotFoundError(
            f"\n[FEHLER] Keine trainierten Gewichte in '{SCRIPT_DIR}' gefunden!\n"
            f"Bitte stelle sicher, dass du zuerst 'main.py' ausführst."
        )

    print(f" -> Weltmodell/Actor geladen aus {SCRIPT_DIR}")
    world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    world_model.eval()
    actor.eval()
    return world_model, actor


# =============================================================================
# 2. FEATURE-EXTRAKTION (LATENT SPACE)
# =============================================================================
@torch.no_grad()
def extract_latents(world_model, episodes):
    """Extrahiert die kombinierten Posterior-Latentzustände (STOCH + DETER)
    sowie die dazugehörigen echten CartPole-Zustände.

    Nutzt den Posterior-Pfad ueber Vektor-Beobachtungen (kein image_encoder):
    Fuer jeden Zeitschritt wird h fortgeschrieben und der Posterior mit der
    naechsten Beobachtung gebildet -- analog zu observe_forward, aber
    deterministisch (mode_one_hot) fuer Reproduzierbarkeit.
    """
    latents_all = []
    true_states_all = []

    for ep in episodes:
        acts = np.stack(ep["actions"])
        if len(acts) < 1:
            continue
        vis = np.stack(ep["vis"]).astype(np.float32)     # (T+1, 4) Beobachtung
        fulls = np.stack(ep["fulls"]).astype(np.float32)  # (T+1, 4) echte Zustaende

        obs = torch.from_numpy(vis).float().unsqueeze(0).to(DEVICE)      # (1, T+1, 4)
        actions = torch.from_numpy(acts).float().unsqueeze(0).to(DEVICE)  # (1, T, 2)
        T = actions.shape[1]

        flat_z, h = world_model.initial(batch_size=1, device=DEVICE)

        for t in range(T):
            a_t = actions[:, t, :]
            h = world_model.gru(world_model.action_stack(torch.cat([flat_z, a_t], dim=-1)), h)

            next_obs = obs[:, t + 1, :]
            posterior_logits = world_model.logits_to_shape(
                world_model.posterior_model(torch.cat([h, next_obs], dim=-1))
            )
            z_post = world_model.mode_one_hot(posterior_logits)
            flat_z = world_model.flatten_latent(z_post)

            feat = torch.cat([flat_z, h], dim=-1)
            latents_all.append(feat.cpu().numpy()[0])
            true_states_all.append(fulls[t + 1])

    return np.stack(latents_all), np.stack(true_states_all)


# =============================================================================
# 3. PYTORCH NN DECODER (DIE SONDE)
# =============================================================================
class LatentDecoderNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def train_and_evaluate_decoder(latents: np.ndarray, true_states: np.ndarray,
                                epochs: int = 60, batch_size: int = 64, lr: float = 1e-3):
    """Trainiert den PyTorch-NN-Decoder (MSE) auf einem Train-Test-Split.

    NMSE = MSE / Var(y)
      0    = Latentvektor erklaert die Variable fast perfekt
      1    = Decoder ist nicht besser als der Mittelwert
    """
    print("\n--- Training des PyTorch NN-Decoders (Latent Space -> 4 State-Werte) ---")

    n_samples = len(latents)
    n_train = int(0.8 * n_samples)

    X_train_raw = torch.tensor(latents[:n_train], dtype=torch.float32)
    y_train_raw = torch.tensor(true_states[:n_train], dtype=torch.float32)
    X_test_raw = torch.tensor(latents[n_train:], dtype=torch.float32)
    y_test_raw = torch.tensor(true_states[n_train:], dtype=torch.float32)

    X_mean = X_train_raw.mean(dim=0, keepdim=True)
    X_std = X_train_raw.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    y_mean = y_train_raw.mean(dim=0, keepdim=True)
    y_std = y_train_raw.std(dim=0, keepdim=True) + 1e-8
    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = LatentDecoderNN(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoche {epoch:02d}/{epochs} | Train-MSE (normalisiert): {epoch_loss / n_train:.6f}")

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(DEVICE)
        y_test_dev = y_test.to(DEVICE)
        test_preds_norm = model(X_test_dev)

        nmse_per_var = ((test_preds_norm - y_test_dev) ** 2).mean(dim=0).cpu().numpy()
        y_std_np = y_std.squeeze().numpy()
        mse_per_var = nmse_per_var * (y_std_np ** 2)
        var_per_var = (y_std_np ** 2)

    print("\n>>> FINALE MSE EVALUIERUNG (Test-Set) <<<")
    print(f"{'Variable':<22}  {'MSE (orig.)':<14}  {'Varianz':<14}  {'NMSE':<8}  Güte")
    print("-" * 72)

    results = {}
    for i, name in enumerate(STATE_NAMES):
        mse = mse_per_var[i]
        var = var_per_var[i]
        nmse = nmse_per_var[i]
        guete = "★★★" if nmse < 0.1 else "★★ " if nmse < 0.3 else "★  " if nmse < 0.7 else "   "
        print(f"  {name:<20}  MSE={mse:.5f}  Var={var:.5f}  NMSE={nmse:.4f}  {guete}")
        results[name] = {"mse": float(mse), "nmse": float(nmse), "var": float(var)}

    return results


# =============================================================================
# 4. VISUALISIERUNGEN
# =============================================================================
def plot_mse_bars(mse_results: dict, save_path: str = "decoder_mse_scores.png"):
    names = list(mse_results.keys())
    mse_vals = [mse_results[n]["mse"] for n in names]
    nmse_vals = [mse_results[n]["nmse"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.barh(names, mse_vals, color="#3498db", edgecolor="white", height=0.5)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("MSE (originale Einheiten)")
    ax1.set_title("Roher MSE\n(nicht direkt vergleichbar!)", fontweight="bold")
    for i, v in enumerate(mse_vals):
        ax1.text(v + max(mse_vals) * 0.01, i, f"{v:.5f}", va="center", fontsize=9)

    colors = ["#2ecc71" if v < 0.1 else "#f39c12" if v < 0.3 else "#e74c3c" for v in nmse_vals]
    ax2.barh(names, nmse_vals, color=colors, edgecolor="white", height=0.5)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.axvline(1.0, color="red", linewidth=1.2, linestyle="--", alpha=0.6, label="NMSE=1 (Mittelwert-Baseline)")
    ax2.set_xlabel("NMSE = MSE / Var(y)   |   0=perfekt, 1=triviale Baseline")
    ax2.set_title("Normalisierter MSE (NMSE)\n(fair vergleichbar zwischen Variablen)", fontweight="bold")
    ax2.legend(fontsize=8)
    for i, v in enumerate(nmse_vals):
        ax2.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    plt.suptitle("Latent-Space Decoder — Zustandsrekonstruktion", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


def plot_pca(latents: np.ndarray, true_states: np.ndarray, save_path: str = "latent_pca.png"):
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(latents)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PCA des Latent Space — Einfärbung nach echten Zuständen", fontsize=14, fontweight="bold")

    for i, (ax, name) in enumerate(zip(axes.flatten(), STATE_NAMES)):
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=true_states[:, i], cmap="RdBu_r", s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


@torch.no_grad()
def plot_reconstructions(world_model, episodes, n_steps: int = 8, save_path: str = "reconstruction.png"):
    """Vergleicht echte Zustandsvektoren mit dem RSSM-Posterior-Decoder-Output.
    (Vektor-Version: statt Bild-Rekonstruktion die 4 Zustandswerte pro Schritt.)"""
    ep = episodes[0]
    vis = np.stack(ep["vis"]).astype(np.float32)
    acts = np.stack(ep["actions"]).astype(np.float32)
    obs = torch.from_numpy(vis).float().unsqueeze(0).to(DEVICE)
    actions = torch.from_numpy(acts).float().unsqueeze(0).to(DEVICE)

    out = world_model.observe_forward(obs, actions)
    recons = out["posterior_predictions"]["observation"][0].cpu().numpy()  # (T, 4)
    real = vis[1:]  # (T, 4), Targets sind die naechsten Beobachtungen
    T_plot = min(n_steps, recons.shape[0], real.shape[0])

    steps = np.arange(1, T_plot + 1)
    n = len(STATE_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    fig.suptitle("Zustands-Rekonstruktion: Echt vs. RSSM-Posterior-Decoder", fontsize=13, fontweight="bold")

    for i, name in enumerate(STATE_NAMES):
        axes[i].plot(steps, real[:T_plot, i], label="Echt", color="black", linewidth=2)
        axes[i].plot(steps, recons[:T_plot, i], label="Rekonstruktion", color="#e67e22", linestyle="--", linewidth=2)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Zeitschritt")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


@torch.no_grad()
def plot_prior_rollout(world_model, episodes, warmup: int = 5, save_path: str = "prior_rollout.png"):
    """Prüft die Halluzinationsfaehigkeit (Prior Rollout) ab Schritt `warmup`.
    (Vektor-Version: vergleicht getraeumte vs. echte Zustandswerte.)"""
    ep = episodes[0]
    if len(ep["actions"]) < warmup + 2:
        print(f" -> Episode zu kurz fuer warmup={warmup}, ueberspringe plot_prior_rollout.")
        return

    vis = np.stack(ep["vis"]).astype(np.float32)
    acts = np.stack(ep["actions"]).astype(np.float32)
    obs = torch.from_numpy(vis).float().unsqueeze(0).to(DEVICE)
    actions = torch.from_numpy(acts).float().unsqueeze(0).to(DEVICE)

    start_flatz, start_h = world_model.posterior_start_state(obs, actions, t0=warmup)
    remaining_actions = actions[:, warmup + 1:, :]
    if remaining_actions.shape[1] == 0:
        print(" -> Keine verbleibenden Schritte nach warmup, ueberspringe plot_prior_rollout.")
        return

    imagined = _open_loop_rollout_fixed_actions(world_model, start_flatz, start_h, remaining_actions)[0].cpu().numpy()
    n_plot = min(8, imagined.shape[0])
    # echte Zukunft ab warmup+2 (analog zum Drift-Helper)
    real_future = vis[warmup + 2: warmup + 2 + n_plot]
    n_plot = min(n_plot, real_future.shape[0])

    steps = np.arange(1, n_plot + 1)
    n = len(STATE_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    fig.suptitle(f"Prior Rollout / Imagination (freies Träumen ab t={warmup})", fontsize=13, fontweight="bold")

    for i, name in enumerate(STATE_NAMES):
        axes[i].plot(steps, real_future[:n_plot, i], label="Echt", color="black", linewidth=2)
        axes[i].plot(steps, imagined[:n_plot, i], label="Geträumt (Prior)", color="#3498db", linestyle="--", linewidth=2)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Schritte nach warmup")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Sprint 5 — Latent-Space-Analyse (Vektor-Modell)")
    print("=" * 70)

    world_model, actor = load_or_train_model()

    env = gym.make("CartPole-v1")
    print("\nSammle 20 frische Analyse-Episoden aus dem Gymnasium-Env...")
    # epsilon=1.0: rein zufaellige Aktionen fuer breite Zustandsraum-Abdeckung.
    episodes = collect_episodes(env, actor, world_model, n_episodes=20, max_steps=150, seed=99, epsilon=1.0)
    env.close()

    print("\nExtrahiere verborgene Zustände aus dem RSSM...")
    latents, true_states = extract_latents(world_model, episodes)
    print(f"  -> Anzahl Datenpunkte: {len(latents)}")
    print(f"  -> Latent-Dimension (Input-Größe für NN): {latents.shape[1]}")

    mse_results = train_and_evaluate_decoder(latents, true_states, epochs=60, batch_size=64, lr=1e-3)

    print("\nGeneriere Analyse-Plots...")
    plot_mse_bars(mse_results, SCRIPT_DIR / "decoder_mse_scores.png")
    plot_pca(latents, true_states, SCRIPT_DIR / "latent_pca.png")
    plot_reconstructions(world_model, episodes, n_steps=8, save_path=SCRIPT_DIR / "reconstruction.png")
    plot_prior_rollout(world_model, episodes, warmup=5, save_path=SCRIPT_DIR / "prior_rollout.png")

    print("\n" + "=" * 70)
    print("Analyse vollständig abgeschlossen. Alle Metriken & Plots aktualisiert.")
    print("=" * 70)