"""
make_gif.py — Erzeugt ein GIF des balancierenden CartPole-Agenten.

Laedt die trainierten Gewichte (world_model.pth, actor.pth) und laesst den
Agenten ueber den kanonischen RSSM-Handel-Pfad spielen (wie evaluate_policy),
rendert dabei jeden Frame und speichert alles als GIF.

Ausfuehren (aus dem Ordner ueber sprint5/):
    python -m cartpole_4D.make_gif
"""
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .models import RSSM, Actor, DEVICE
from .data import visible_state, onehot_action

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_font(size=26):
    """Versucht eine TrueType-Font zu laden; faellt auf die PIL-Standardfont
    zurueck, falls keine gefunden wird (laeuft so auf jedem System)."""
    for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _annotate_frame(frame, step, episode=None, font=None):
    """Schreibt einen Schrittzaehler (und optional die Episode) oben auf den Frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"Schritt: {step}"
    if episode is not None:
        text = f"Episode {episode}   " + text
    # halbtransparenter Hintergrund hinter dem Text fuer Lesbarkeit
    x, y = 15, 12
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def make_cartpole_gif(gif_path=None, max_steps=500, seed=0, fps=30,
                      greedy=True, n_episodes=1):
    """Erzeugt ein GIF des balancierenden CartPole.

    Args:
        gif_path:   Zielpfad (default: sprint5/cartpole_balance.gif)
        max_steps:  max. Schritte pro Episode (CartPole-v1 endet spaetestens bei 500)
        seed:       Startseed der Umgebung
        fps:        Bilder pro Sekunde im GIF
        greedy:     True = beste Aktion (argmax), False = gesampelt
        n_episodes: mehrere Episoden hintereinander ins selbe GIF
    """
    if gif_path is None:
        gif_path = SCRIPT_DIR / "cartpole_balance.gif"

    # --- Modelle laden ---
    wm = RSSM().to(DEVICE)
    feat = wm.stoch_size + wm.h_size
    actor = Actor(feat).to(DEVICE)
    wm_path = SCRIPT_DIR / "world_model.pth"
    actor_path = SCRIPT_DIR / "actor.pth"
    if not (wm_path.exists() and actor_path.exists()):
        raise FileNotFoundError(
            f"\n[FEHLER] Keine Gewichte in '{SCRIPT_DIR}' gefunden!\n"
            f"Bitte zuerst 'python -m sprint5.main' ausfuehren."
        )
    wm.load_state_dict(torch.load(wm_path, map_location=DEVICE))
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    wm.eval(); actor.eval()
    print(f" -> Gewichte geladen aus {SCRIPT_DIR}")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    font = _load_font(size=26)
    show_episode = n_episodes > 1   # Episode nur anzeigen, wenn es mehrere gibt
    frames = []
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # RSSM-Zustand mitfuehren (kanonischer Handel, wie evaluate_policy)
        flatz, h = wm.initial(batch_size=1, device=DEVICE)
        prev_action = torch.zeros(1, wm.action_size, device=DEVICE)
        obs_t = torch.tensor(visible_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        flatz, h, feat_v = wm.act_step(flatz, h, prev_action, obs_t)

        done = False
        steps = 0
        while not done and steps < max_steps:
            ep_num = (ep + 1) if show_episode else None
            frames.append(_annotate_frame(env.render(), steps, ep_num, font))
            with torch.no_grad():
                logits = actor(feat_v)
                if greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    action = torch.distributions.Categorical(logits=logits).sample().item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            prev_action = torch.tensor(onehot_action(action), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            obs_t = torch.tensor(visible_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flatz, h, feat_v = wm.act_step(flatz, h, prev_action, obs_t)

        # letzter Frame mit finalem Schrittzaehler
        ep_num = (ep + 1) if show_episode else None
        frames.append(_annotate_frame(env.render(), steps, ep_num, font))
        total_steps += steps
        print(f"   Episode {ep+1}: {steps} Schritte balanciert")

    env.close()

    # --- GIF speichern ---
    # duration = Sekunden pro Frame; loop=0 -> endlos
    imageio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)
    print(f"\n -> GIF gespeichert: {gif_path}")
    print(f"    {len(frames)} Frames, {total_steps} Schritte gesamt")
    return gif_path


if __name__ == "__main__":
    make_cartpole_gif()