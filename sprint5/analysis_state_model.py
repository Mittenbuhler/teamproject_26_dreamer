"""
analysis.py — Sprint 5
Analyse-Skript für das STATE-basierte Dreamer-Modell.

Ausführen mit: .venv/bin/python -m sprint5.analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path

from .models import RSSM, Actor, Critic, ObsActor, DEVICE
from .data import SequenceDataset, collect_episodes, visible_state

VAR_NAMES = ["Position (x)", "Winkel (θ)"]
SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Setup / Modell-Loader
# =============================================================================
def load_or_train():
    world_model = RSSM().to(DEVICE)
    feat_size = world_model.stoch_size + world_model.h_size
    actor = Actor(feat_size).to(DEVICE)
    critic = Critic(feat_size).to(DEVICE)

    wm_path = SCRIPT_DIR / "world_model.pth"
    actor_path = SCRIPT_DIR / "actor.pth"
    critic_path = SCRIPT_DIR / "critic.pth"

    if not (wm_path.exists() and actor_path.exists()):
        raise FileNotFoundError(
            f"\n[FEHLER] Keine trainierten Gewichte in '{SCRIPT_DIR}' gefunden!\n"
            f"Bitte stelle sicher, dass du zuerst 'main.py' ausführst."
        )

    print(f"Lade trainierte Gewichte aus {SCRIPT_DIR} ...")
    world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    if critic_path.exists():
        critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))

    obs_actor = ObsActor(world_model.obs_size, feat_size, policy=actor).to(DEVICE)
    return world_model, actor, critic, obs_actor


# =============================================================================
# 1. Weltmodell-Genauigkeit (Posterior vs. Prior)
# =============================================================================
@torch.no_grad()
def analyze_world_model_accuracy(world_model, episodes, seq_len: int = 16, batch_size: int = 16):
    world_model.eval()
    dataset = SequenceDataset(episodes, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError("Keine ausreichend langen Test-Episoden vorhanden.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    post_obs_errs, prior_obs_errs = [], []
    post_reward_errs, prior_reward_errs = [], []
    post_cont_correct, prior_cont_correct, n_cont = 0, 0, 0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = world_model.observe_forward(batch["observations"], batch["actions"])
        target_obs = batch["observations"][:, 1:, :]

        post_obs_errs.append(((out["posterior_predictions"]["observation"] - target_obs) ** 2).mean(dim=(0, 1)).cpu().numpy())
        prior_obs_errs.append(((out["prior_predictions"]["observation"] - target_obs) ** 2).mean(dim=(0, 1)).cpu().numpy())
        post_reward_errs.append(F.mse_loss(out["posterior_predictions"]["reward"], batch["rewards"]).item())
        prior_reward_errs.append(F.mse_loss(out["prior_predictions"]["reward"], batch["rewards"]).item())

        post_pred_cont = (out["posterior_predictions"]["continuelogit"].sigmoid() > 0.5).float()
        prior_pred_cont = (out["prior_predictions"]["continuelogit"].sigmoid() > 0.5).float()
        post_cont_correct += (post_pred_cont == batch["continues"]).sum().item()
        prior_cont_correct += (prior_pred_cont == batch["continues"]).sum().item()
        n_cont += batch["continues"].numel()

    return {
        "obs_mse_posterior": np.mean(post_obs_errs, axis=0),
        "obs_mse_prior":     np.mean(prior_obs_errs, axis=0),
        "reward_mse_posterior": float(np.mean(post_reward_errs)),
        "reward_mse_prior":     float(np.mean(prior_reward_errs)),
        "continue_acc_posterior": post_cont_correct / n_cont,
        "continue_acc_prior":     prior_cont_correct / n_cont,
    }


def plot_world_model_accuracy(results: dict, save_path: str):
    labels = VAR_NAMES + ["Reward"]
    post_vals = list(results["obs_mse_posterior"]) + [results["reward_mse_posterior"]]
    prior_vals = list(results["obs_mse_prior"]) + [results["reward_mse_prior"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x - width / 2, post_vals, width, label="Posterior (mit Observation)", color="#2ecc71")
    axes[0].bar(x + width / 2, prior_vals, width, label="Prior (rein träumend)", color="#e67e22")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Vorhersagefehler", fontweight="bold")
    axes[0].legend()

    accs = [results["continue_acc_posterior"], results["continue_acc_prior"]]
    axes[1].bar(["Posterior", "Prior"], accs, color=["#2ecc71", "#e67e22"])
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Zufall")
    axes[1].set_title("Continue-Signal (Genauigkeit)", fontweight="bold")
    axes[1].legend()

    plt.suptitle("Analyse 1: Weltmodell-Genauigkeit", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# =============================================================================
# 2. Balance-Dauer
# =============================================================================
def analyze_balance_duration(obs_actor, n_episodes: int = 50, max_steps: int = 200):
    env = gym.make("CartPole-v1")
    
    def rollout(policy_fn):
        lengths = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            steps, done = 0, False
            while not done and steps < max_steps:
                action = policy_fn(obs)
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                steps += 1
            lengths.append(steps)
        return np.array(lengths)

    trained = rollout(lambda o: torch.distributions.Categorical(logits=obs_actor(torch.tensor(visible_state(o), dtype=torch.float32).unsqueeze(0))).sample().item())
    random_ = rollout(lambda o: env.action_space.sample())
    env.close()
    return {"trained": trained, "random": random_}


def plot_balance_duration(results: dict, save_path: str):
    trained, random_ = results["trained"], results["random"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 210, 10)
    ax.hist(random_, bins=bins, alpha=0.5, label=f"Zufall (Ø={random_.mean():.1f})", color="#95a5a6")
    ax.hist(trained, bins=bins, alpha=0.6, label=f"Actor (Ø={trained.mean():.1f})", color="#2ecc71")
    ax.set_xlabel("Schritte überlebt")
    ax.set_ylabel("Episoden Anzahl")
    ax.set_title("Analyse 2: Kontroll-Langlebigkeit", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# =============================================================================
# 3. Imagination-Horizont-Drift & Hilfsfunktion
# =============================================================================
@torch.no_grad()
def _open_loop_rollout_fixed_actions(world_model, start_flatz, start_h, real_actions):
    flatz, h = start_flatz, start_h
    obs_preds = []
    for t in range(real_actions.shape[1]):
        act_t = real_actions[:, t, :]
        h = world_model.gru(world_model.action_stack(torch.cat([flatz, act_t], dim=-1)), h)
        prior_logits = world_model.logits_to_shape(world_model.prior_model(h))
        flatz = world_model.flatten_latent(world_model.mode_one_hot(prior_logits))
        pred = world_model.prediction_heads(flatz, h)
        obs_preds.append(pred["observation"])
    return torch.stack(obs_preds, dim=1)


@torch.no_grad()
def analyze_imagination_drift(world_model, episodes, horizon: int = 15):
    world_model.eval()
    usable = [ep for ep in episodes if len(ep["actions"]) >= horizon + 1][:20]
    per_step_errors = []
    
    for ep in usable:
        vis = np.array(ep["vis"], dtype=np.float32)
        acts = np.array(ep["actions"], dtype=np.float32)
        o = torch.tensor(vis, device=DEVICE).unsqueeze(0)
        a = torch.tensor(acts, device=DEVICE).unsqueeze(0)

        start_flatz, start_h = world_model.posterior_start_state(o, a, t0=0)
        real_actions = a[:, 1:1 + horizon, :]
        imagined_obs = _open_loop_rollout_fixed_actions(world_model, start_flatz, start_h, real_actions)[0]

        real_future = vis[2:2 + horizon]
        T_valid = min(len(real_future), imagined_obs.shape[0])
        err = ((imagined_obs[:T_valid].cpu().numpy() - real_future[:T_valid]) ** 2).mean(axis=-1)
        per_step_errors.append(err)

    padded = np.full((len(per_step_errors), horizon), np.nan)
    for i, e in enumerate(per_step_errors):
        padded[i, :len(e)] = e
    return {"mean": np.nanmean(padded, axis=0), "std": np.nanstd(padded, axis=0)}


def plot_imagination_drift(results: dict, save_path: str):
    mean_err, std_err = results["mean"], results["std"]
    steps = np.arange(1, len(mean_err) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, mean_err, color="#3498db", marker="o", label="Mittlerer Drift-MSE")
    ax.fill_between(steps, mean_err - std_err, mean_err + std_err, color="#3498db", alpha=0.2)
    ax.set_xlabel("Schritte in die Zukunft geträumt")
    ax.set_ylabel("MSE (Zustand)")
    ax.set_title("Analyse 3: Akkumulierter Dynamik-Drift im Open-Loop", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# =============================================================================
# NEU! 4. "Traum vs. Realität" (Direkter Trajektorien-Vergleich)
# =============================================================================
@torch.no_grad()
def plot_qualitative_trajectory(world_model, test_episodes, save_path: str, horizon: int = 25):
    """ Pickt eine Episode und stellt echten Zustand vs. geträumten Zustand dar """
    world_model.eval()
    # Finde die längste Episode für einen schönen Plot
    ep = max(test_episodes, key=lambda e: len(e["actions"]))
    H = min(horizon, len(ep["actions"]) - 2)

    vis = np.array(ep["vis"], dtype=np.float32)
    acts = np.array(ep["actions"], dtype=np.float32)

    o = torch.tensor(vis, device=DEVICE).unsqueeze(0)
    a = torch.tensor(acts, device=DEVICE).unsqueeze(0)

    # Initialisiere Zustand im Weltmodell aus erstem Schritt
    start_flatz, start_h = world_model.posterior_start_state(o, a, t0=0)
    real_actions = a[:, 1:1+H, :]
    
    # Generiere davorhergesagte Flugbahn im Geiste des Modells
    imagined_obs = _open_loop_rollout_fixed_actions(world_model, start_flatz, start_h, real_actions)[0].cpu().numpy()
    real_future = vis[2:2+H]

    steps = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, name in enumerate(VAR_NAMES):
        axes[i].plot(steps, real_future[:, i], label="Reale Umgebung (Wahrheit)", color="black", linewidth=2.5)
        axes[i].plot(steps, imagined_obs[:, i], label="Modell-Traum (Prior)", color="#3498db", linestyle="--", linewidth=2.5)
        axes[i].set_title(f"Zustandsverlauf: {name}", fontweight="bold")
        axes[i].set_xlabel("Zeitschritte voraus")
        axes[i].set_ylabel("Skalierter Wert")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.suptitle("Analyse 4: Intuitions-Check — Wie präzise 'träumt' das Weltmodell?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


# =============================================================================
# NEU! 5. Policy-Entscheidungslandschaft (2D-Heatmap)
# =============================================================================
@torch.no_grad()
def plot_policy_landscape(obs_actor, save_path: str):
    """ Scannt das Netzwerk auf einem 2D-Gitter ab, um die Logik zu visualisieren """
    obs_actor.eval()
    
    # Erstelle dichte Auflösung für Position x und Polwinkel theta
    x_grid = np.linspace(-2.0, 2.0, 100)
    theta_grid = np.linspace(-0.22, 0.22, 100)
    X, Y = np.meshgrid(x_grid, theta_grid)
    
    # Zusammenbauen als [Position, Winkel]-Eingabe
    grid_data = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    grid_tensor = torch.tensor(grid_data, device=DEVICE)
    
    logits = obs_actor(grid_tensor)
    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # Wahrscheinlichkeit für Aktion 1 ("RECHTS")
    Z = probs.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # RdYlGn: Rot = Drücke nach Links, Grün = Drücke nach Rechts
    contour = ax.pcolormesh(X, Y, Z, cmap="RdYlGn", vmin=0, vmax=1, shading='auto', alpha=0.85)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Aktions-Wahrscheinlichkeit für RECHTS (Aktion 1)", fontsize=10)
    
    ax.set_xlabel("Cart-Position (x)")
    ax.set_ylabel("Pol-Winkel (θ) in Radian")
    ax.set_title("Analyse 5: Gelerntes Regelwerk des Actors (Policy-Map)", fontweight="bold", fontsize=12)
    
    # Null-Linien zur Orientierung einzeichnen
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.axvline(0, color="black", linewidth=1, linestyle=":")
    
    # Erläuterungsbox
    ax.text(-1.9, 0.16, "Stange kippt nach Links\n-> Drücke LINKS (Rot)", color="darkred", bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    ax.text(0.3, -0.19, "Stange kippt nach Rechts\n-> Drücke RECHTS (Grün)", color="darkgreen", bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Sprint 5 — Fortgeschrittene Konzept-Analyse des Dreamer-Modells")
    print("=" * 70)

    world_model, actor, critic, obs_actor = load_or_train()

    env = gym.make("CartPole-v1")
    print("\nSammle frische Test-Episoden aus der echten Umgebung ...")
    test_episodes = collect_episodes(env, obs_actor, n_episodes=30, max_steps=200, seed=999, epsilon=0.0)
    env.close()

    # Bestandsanalysen
    print("\n[1/5] Berechne Weltmodell-Genauigkeiten ...")
    wm_results = analyze_world_model_accuracy(world_model, test_episodes)
    plot_world_model_accuracy(wm_results, SCRIPT_DIR / "wm_accuracy.png")

    print("[2/5] Führe Benchmark-Rollouts durch (Trained vs Random) ...")
    duration_results = analyze_balance_duration(obs_actor, n_episodes=50)
    plot_balance_duration(duration_results, SCRIPT_DIR / "balance_duration.png")

    print("[3/5] Ermittle Imagination-Drift über Zeithorizont ...")
    drift_results = analyze_imagination_drift(world_model, test_episodes, horizon=15)
    plot_imagination_drift(drift_results, SCRIPT_DIR / "imagination_drift.png")

    # Neue Analysen
    print("[4/5] Generiere qualitativen 'Traum vs. Realität'-Vergleich ...")
    plot_qualitative_trajectory(world_model, test_episodes, SCRIPT_DIR / "trajectory_comparison.png", horizon=25)

    print("[5/5] Erzeuge 2D-Policy-Entscheidungslandschaft ...")
    plot_policy_landscape(obs_actor, SCRIPT_DIR / "policy_landscape.png")

    print("\n" + "=" * 70)
    print("Analyse abgeschlossen!")
    print("=" * 70)