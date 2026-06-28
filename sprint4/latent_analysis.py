"""
latent_analysis.py — Sprint 4 (Restrukturiert)
Latent-Space-Analyse des trainierten Dreamer-Modells mittels PyTorch NN-Decoder.

Zentraler Fokus:
  Ein PyTorch-basiertes Neuronales Netzwerk (nn.Linear) decodiert den 
  Latent Space (z & h) direkt in die 4 echten CartPole-Zustandsvariablen 
  und evaluiert die Genauigkeit anhand des Mean Squared Error (MSE).

Struktur des Files:
  1. Setup & Hilfsfunktionen (Modell laden, Episoden sammeln)
  2. Feature-Extraktion (Posterior-Zustände aus RSSM extrahieren)
  3. PyTorch NN Decoder (Modelldefinition, Training & MSE-Evaluierung)
  4. Visualisierungen (MSE-Scores, PCA, Rekonstruktion, Prior-Rollout)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Importe aus dem existierenden Projekt
from RSSM_cartpole import RSSM, DEVICE, RESOLUTION, STOCH_SIZE, DETER_SIZE
from CartPole import collect_episodes_image

# Namen der 4 CartPole-Zustandsvariablen
STATE_NAMES = ["Position (x)", "Geschw. (ẋ)", "Winkel (θ)", "Winkelgesch. (θ̇)"]


# =============================================================================
# 1. SETUP & HILFSFUNKTIONEN
# =============================================================================

def load_or_train_model(weights_path: str = "dreamer_sprint4.pt") -> RSSM:
    """Lädt ein vortrainiertes RSSM-Modell oder startet das Training neu."""
    from RSSM_cartpole import train
    model = RSSM().to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f" -> Weltmodell erfolgreich geladen: {weights_path}")
    else:
        print(f" -> {weights_path} nicht gefunden — Starte automatisches Training...")
        model = train(n_iterations=20, n_seed_eps=15, n_new_eps=4)
    model.eval()
    return model


# =============================================================================
# 2. FEATURE-EXTRAKTION (LATENT SPACES)
# =============================================================================

@torch.no_grad()
def extract_latents(model: RSSM, episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrahiert die kombinierten Posterior-Latentzustände (STOCH + DETER) 
    sowie die dazugehörigen echten CartPole-Zustände aus den Episoden.
    """
    latents_all = []
    true_states_all = []

    for ep in episodes:
        frames = torch.from_numpy(np.stack(ep["frames"])[:, np.newaxis, :, :]).float().unsqueeze(0).to(DEVICE)
        actions = torch.from_numpy(np.stack(ep["actions"])).float().unsqueeze(0).to(DEVICE)

        B, Tp1, C1, H, W = frames.shape
        T = actions.shape[1]

        # Vorkalkulation aller Encoder-Embeddings für Effizienz
        enc_all = model.encoder(frames.view(B * Tp1, C1, H, W)).view(B, Tp1, -1)

        flat_z = torch.zeros(B, STOCH_SIZE, device=DEVICE)
        h = torch.zeros(B, DETER_SIZE, device=DEVICE)
        full_states = np.stack(ep["full_states"])

        for t in range(T):
            a_t = actions[:, t, :]
            inp = model.action_stack(torch.cat([flat_z, a_t], dim=-1))
            h = model.gru(inp, h)
            
            e_next = enc_all[:, t + 1, :]
            post_logits = model._to_categorical(model.posterior_net(torch.cat([h, e_next], dim=-1)))
            z_post = model._straight_through(post_logits)
            flat_z = model._flatten(z_post)
            
            # Feature-Vektor repräsentiert den vollständigen Latent Space (Dim: 64 + 256 = 320)
            feat = torch.cat([flat_z, h], dim=-1)

            latents_all.append(feat.cpu().numpy()[0])
            true_states_all.append(full_states[t + 1])

    return np.stack(latents_all), np.stack(true_states_all)


# =============================================================================
# 3. PYTORCH NN DECODER (DIE SONDE)
# =============================================================================

class LatentDecoderNN(nn.Module):
    """
    Ein neuronales Netzwerk zur Decodierung des Latent-Spaces in CartPole-Zustände.
    Nutzt standardmäßig ein lineares Layer (nn.Linear) analog zu einer Linearen Probe.
    """
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        # Mappt den Latent Space (z.B. 320 Dimensionen) auf die 4 CartPole-Werte
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def train_and_evaluate_decoder(latents: np.ndarray, true_states: np.ndarray, 
                               epochs: int = 60, batch_size: int = 64, lr: float = 1e-3):
    """
    Trainiert den PyTorch-NN-Decoder mittels MSE-Verlust auf einem Train-Test-Split
    und gibt den finalen MSE für jede einzelne Zustandsvariable aus.
    """
    print("\n--- Training des PyTorch NN-Decoders (Latent Space -> 4 State-Werte) ---")
    
    # Train-Test-Split (80% / 20%)
    n_samples = len(latents)
    n_train = int(0.8 * n_samples)
    
    X_train_raw = torch.tensor(latents[:n_train], dtype=torch.float32)
    y_train = torch.tensor(true_states[:n_train], dtype=torch.float32)
    X_test_raw = torch.tensor(latents[n_train:], dtype=torch.float32)
    y_test = torch.tensor(true_states[n_train:], dtype=torch.float32)

    # Standardisierung der Eingangs-Features für stabileres NN-Training
    X_mean = X_train_raw.mean(dim=0, keepdim=True)
    X_std = X_train_raw.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # Daten-Loader vorbereiten
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modell, Optimizer und Loss initialisieren
    input_dim = X_train.shape[1]
    model = LatentDecoderNN(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Wie gewünscht: Evaluation via Mean Squared Error

    # Trainings-Loop
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
            print(f"  Epoche {epoch:02d}/{epochs} | Train-MSE: {epoch_loss / n_train:.6f}")

    # Evaluierung auf dem Test-Set
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        test_preds = model(X_test)
        
        total_test_mse = criterion(test_preds, y_test).item()
        # Einzelne MSEs pro Variable berechnen
        mse_per_variable = ((test_preds - y_test) ** 2).mean(dim=0).cpu().numpy()

    print("\n>>> FINALE MSE EVALUIERUNG (Test-Set) <<<")
    print(f"  Gesamt-MSE über alle Variablen: {total_test_mse:.6f}")
    print("-" * 50)
    
    mse_dict = {}
    for i, name in enumerate(STATE_NAMES):
        var_mse = mse_per_variable[i]
        mse_dict[name] = var_mse
        # Visueller Balken im Terminal (je kürzer, desto besser!)
        bar_len = max(1, int((1.0 - min(var_mse, 1.0)) * 30))
        bar = "█" * bar_len
        print(f"  {name:<22} | MSE = {var_mse:.6f}   {bar}")
        
    return mse_dict


# =============================================================================
# 4. VISUALISIERUNGS-FUNKTIONEN
# =============================================================================

def plot_mse_bars(mse_results: dict, save_path: str = "decoder_mse_scores.png"):
    """Erstellt ein Balkendiagramm der MSE-Fehler für die 4 Variablen."""
    names = list(mse_results.keys())
    mse_vals = list(mse_results.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, mse_vals, color="#3498db", edgecolor="white", height=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Squared Error (MSE) - Je niedriger desto besser")
    ax.set_title("CartPole-Zustandsrekonstruktion via PyTorch NN-Decoder", fontweight="bold")

    for bar, v in zip(bars, mse_vals):
        ax.text(v + (max(mse_vals)*0.01), bar.get_y() + bar.get_height() / 2,
                f"{v:.5f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


def plot_pca(latents: np.ndarray, true_states: np.ndarray, save_path: str = "latent_pca.png"):
    """2D-PCA des Latentbereichs, eingefärbt nach den wahren CartPole-Werten."""
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
def plot_reconstructions(model: RSSM, episodes: list[dict], n_steps: int = 8, save_path: str = "reconstruction.png"):
    """Visualisiert den direkten Vergleich zwischen echten Frames und dem RSSM-Decoder-Output."""
    ep = episodes[0]
    frames = torch.from_numpy(np.stack(ep["frames"])[:, np.newaxis, :, :]).float().unsqueeze(0).to(DEVICE)
    actions = torch.from_numpy(np.stack(ep["actions"])).float().unsqueeze(0).to(DEVICE)

    out = model.observe_forward(frames, actions)
    recons = out["reconstructions"][0].cpu()
    T_plot = min(n_steps, recons.shape[0])

    fig, axes = plt.subplots(3, T_plot, figsize=(2.2 * T_plot, 6))
    fig.suptitle("Bild-Rekonstruktion: Echt vs. RSSM-Decoder vs. Absoluter Fehler", fontsize=13, fontweight="bold")

    for t in range(T_plot):
        real = frames[0, t + 1, 0].cpu().numpy()
        recon = recons[t, 0].numpy()
        err = np.abs(real - recon)

        axes[0, t].imshow(real, cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(recon, cmap="gray", vmin=0, vmax=1)
        im = axes[2, t].imshow(err, cmap="hot", vmin=0, vmax=0.5)

        axes[0, t].set_title(f"t={t+1}", fontsize=9)
        for ax in axes[:, t]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


@torch.no_grad()
def plot_prior_rollout(model: RSSM, episodes: list[dict], warmup: int = 5, save_path: str = "prior_rollout.png"):
    """Überprüft die Halluzinationsfähigkeit (Prior Rollout) ohne visuelle Inputs ab Schritt X."""
    ep = episodes[0]
    frames = torch.from_numpy(np.stack(ep["frames"])[:, np.newaxis, :, :]).float().unsqueeze(0).to(DEVICE)
    actions = torch.from_numpy(np.stack(ep["actions"])).float().unsqueeze(0).to(DEVICE)

    imagined = model.prior_rollout(frames, actions, warmup_steps=warmup)[0].cpu()
    n_plot = min(8, imagined.shape[0])

    fig, axes = plt.subplots(2, n_plot, figsize=(2.2 * n_plot, 5))
    fig.suptitle(f"Prior Rollout / Imagination (freies Träumen ab t={warmup})", fontsize=13, fontweight="bold")

    for i in range(n_plot):
        t = warmup + i
        real = frames[0, t + 1, 0].cpu().numpy()
        imag = imagined[i, 0].numpy()

        axes[0, i].imshow(real, cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(imag, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={t+1}", fontsize=9)
        for ax in axes[:, i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" -> Grafik gespeichert: {save_path}")


# =============================================================================
# 5. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sprint 4 — Restrukturierte Latent Space Analyse (PyTorch NN)")
    print("=" * 70)

    # 1. Weltmodell laden oder frisch trainieren
    model = load_or_train_model("dreamer_sprint4.pt")

    # 2. Test-Episoden aus dem Environment generieren
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    print("\nSammle 20 frische Analyse-Episoden aus dem Gymnasium-Env...")
    episodes = collect_episodes_image(env, n_episodes=20, max_steps=150, resolution=RESOLUTION, seed=99)
    env.close()

    # 3. Features extrahieren (Kombination aus h und z)
    print("\nExtrahiere verborgene Zustände aus dem RSSM...")
    latents, true_states = extract_latents(model, episodes)
    print(f"  -> Anzahl Datenpunkte: {len(latents)}")
    print(f"  -> Latent-Dimension (Input-Größe für NN): {latents.shape[1]}")

    # 4. PyTorch NN-Decoder trainieren und Fehler als MSE ausgeben
    mse_results = train_and_evaluate_decoder(latents, true_states, epochs=60, batch_size=64, lr=1e-3)

    # 5. Generierung der Plots
    print("\nGeneriere Analyse-Plots...")
    plot_mse_bars(mse_results)
    plot_pca(latents, true_states)
    plot_reconstructions(model, episodes, n_steps=8)
    plot_prior_rollout(model, episodes, warmup=5)

    print("\n" + "=" * 70)
    print("Analyse vollständig abgeschlossen. Alle Metriken & Plots aktualisiert.")
    print("=" * 70)