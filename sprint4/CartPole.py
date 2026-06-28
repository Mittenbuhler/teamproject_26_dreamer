import gymnasium as gym
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_frame(env, resolution=32):
    """
    Rendert den aktuellen CartPole-Frame als Graustufenbild.
    Rückgabe: float32-Array der Form (resolution, resolution), Werte in [0, 1].
    """
    frame = env.render()
    img = Image.fromarray(frame)
    img = img.convert("L")
    img = img.resize((resolution, resolution))
    return np.array(img, dtype=np.float32) / 255.0



def collect_episodes_image(
    env,
    n_episodes: int = 10,
    max_steps: int = 200,
    resolution: int = 32,
    seed: int | None = None,
) -> list[dict]:
    """
    Sammelt zufällige CartPole-Episoden mit Bildinput.
 
    Jede Episode enthält:
    - frames:     (T+1, resolution, resolution)  float32 Pixel in [0,1]
    - full_states:(T+1, 4)                        float32 [pos, vel, angle, ang_vel]
    - actions:    (T, 2)                          float32 one-hot
    - rewards:    (T, 1)                          float32
    - continues:  (T, 1)                          float32  1=weiter, 0=Ende
    """
    episodes = []
    for ep_idx in range(n_episodes):
        if seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep_idx)
 
        ep = {
            "frames": [],
            "full_states": [],
            "actions": [],
            "rewards": [],
            "continues": [],
        }
        done = False
        steps = 0
 
        while not done and steps < max_steps:
            frame = get_frame(env, resolution)
            a = env.action_space.sample()
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
 
            one_hot = np.zeros(2, dtype=np.float32)
            one_hot[a] = 1.0
 
            ep["frames"].append(frame)
            ep["full_states"].append(obs.astype(np.float32))
            ep["actions"].append(one_hot)
            ep["rewards"].append([float(r)])
            ep["continues"].append([0.0 if done else 1.0])
 
            obs = next_obs
            steps += 1
 
        # Letztes Frame + Zustand anhängen (T+1 Observationen für T Aktionen)
        ep["frames"].append(get_frame(env, resolution))
        ep["full_states"].append(obs.astype(np.float32))
        episodes.append(ep)
 
    return episodes
 
 
# ---------------------------------------------------------------------------
# Schnelltest & Visualisierung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RESOLUTION = 32
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
 
    frame = get_frame(env, RESOLUTION)
    print(f"Frame shape: {frame.shape}, min={frame.min():.3f}, max={frame.max():.3f}")
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
 
    # 32x32 Graustufen
    axes[0].imshow(frame, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Graustufen {RESOLUTION}x{RESOLUTION}")
    axes[0].axis("off")
 
    # 64x64 zum Vergleich
    frame64 = get_frame(env, 64)
    axes[1].imshow(frame64, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Graustufen 64x64")
    axes[1].axis("off")
 
    # Originalframe
    axes[2].imshow(env.render())
    axes[2].set_title("RGB Original")
    axes[2].axis("off")
 
    plt.suptitle(f"CartPole Bildinput — Zustand: {obs.round(3)}", fontsize=11)
    plt.tight_layout()
    plt.savefig("cartpole_frames.png", dpi=100)
    plt.show()
    print("Gespeichert: cartpole_frames.png")
    env.close()
