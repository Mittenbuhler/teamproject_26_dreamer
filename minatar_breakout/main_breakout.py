"""
main_breakout.py — Trainings-Orchester fuer Dreamer auf MinAtar Breakout.

Verdrahtet Weltmodell (CNN-RSSM), Actor und Critic und trainiert sie iterativ:
  1. Zufallsdaten sammeln -> Weltmodell vortrainieren
  2. Actor-Critic-Vortraining in der Imagination
  3. Hauptschleife: Daten sammeln -> Weltmodell -> Actor-Critic -> evaluieren

Gegenueber der CartPole-Version (main.py) sind nur die Umgebung (native MinAtar
statt Gym) und die Beobachtung (10x10x4-Bild statt 4D-Vektor) anders. Der
Lern-Kern -- lambda-Returns, Advantage-Normalisierung, Critic-Normalisierung,
Best-Modell-Speicherung, Entropie-Annealing -- ist identisch, weil er rein im
Latent-Raum des Weltmodells arbeitet.

Ausfuehren mit:  python -m sprint5.main_breakout
"""
import torch
import torch.optim as optim
from dataclasses import dataclass
# [BREAKOUT] Breakout-Module (CNN-RSSM, MinAtar-Daten, Bild-Loss)
from .models_breakout import RSSM, Actor, Critic, DEVICE, ACTION_SIZE
from .data_breakout import (ReplayBuffer, collect_episodes, make_env,
                            state_to_obs, onehot_action)
from .train_breakout import train_world_model, train_actor_critic


@torch.no_grad()
def evaluate_policy(env_unused, world_model, actor, episodes=5, max_steps=1000, game="breakout"):
    """[BREAKOUT] Evaluiert die Policy auf MinAtar Breakout (native API, greedy).
    Return = summierter Reward (Anzahl zerstoerter Bricks) pro Episode."""
    world_model.eval()
    actor.eval()
    returns = []
    for ep_idx in range(episodes):
        env = make_env(game, seed=1000 + ep_idx)
        action_map = env.minimal_action_set()
        env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        flatz, h = world_model.initial(batch_size=1, device=DEVICE)
        prev_action = torch.zeros(1, ACTION_SIZE, device=DEVICE)
        obs_t = torch.tensor(state_to_obs(env.state()), device=DEVICE).unsqueeze(0)
        flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)

        while not done and steps < max_steps:
            logits = actor(feat)
            a_idx = torch.distributions.Categorical(logits=logits).sample().item()
            reward, done = env.act(action_map[a_idx])
            total_reward += reward
            steps += 1

            prev_action = torch.tensor(onehot_action(a_idx), device=DEVICE).unsqueeze(0)
            obs_t = torch.tensor(state_to_obs(env.state()), device=DEVICE).unsqueeze(0)
            flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)
        returns.append(total_reward)
    return sum(returns) / len(returns)


LR_WORLD = 2e-4   # DreamerV2 world model lr (Tab. D.1)
LR_ACTOR = 4e-5   # DreamerV2 actor lr
LR_CRITIC = 1e-4  # DreamerV2 critic lr


@dataclass
class TrainConfig:
    """Hyperparameter fuer das Breakout-Dreamer-Training.

    Der eval_return bei Breakout ist die Anzahl zerstoerter Bricks pro Episode
    (nicht ueberlebte Schritte wie bei CartPole). Breakout lernt deutlich
    langsamer als CartPole -- rechne mit vielen hundert Iterationen und Stunden
    Rechenzeit, bis der Agent zuverlaessig Bricks trifft.
    """
    # --- Trainingsdauer & Datenmenge ---
    iterations: int = 200                  # Haupt-Trainingsschleife (fuer Breakout eher mehr)
    epochs_per_phase: int = 4              # Weltmodell-Updates pro Iteration
    batch_size: int = 32
    seq_len: int = 25                      # Sequenzlaenge fuer das Weltmodell-Training
    replay_capacity: int = 2000            # Replay-Buffer-Groesse (in Episoden)
    imagination_horizon: int = 15          # wie weit der Actor "traeumt"

    # --- Exploration via Entropie-Annealing ---
    # Hoch starten (breite Exploration -> findet zuverlaessiger ueber verschiedene
    # Seeds den Weg nach oben), dann langsam senken (verfeinert die Policy). Der
    # actor_entropy_coeff wird im Loop linear zwischen Start und Ende interpoliert.
    # Hinweis: Breakout hat 3 Aktionen -> uniforme Entropie ist ln(3) ~ 1.10.
    actor_entropy_coeff: float = 3e-3      # Fallback-Startwert (ueberschrieben durch entropy_start)
    entropy_start: float = 6e-3            # viel Exploration am Anfang
    entropy_end: float = 1e-3              # DreamerV2-Wert gegen Ende (Tab. D.1)

    # --- Aufwaermphase (vor dem Haupttraining) ---
    warmup_steps: int = 5                  # Posterior-Warmup vor der Imagination
    initial_random_episodes: int = 25      # Zufallsdaten zum Start -> breite Zustandsabdeckung
    initial_world_model_epochs: int = 30   # Weltmodell vortrainieren, bevor der Actor lernt
    initial_ac_pretrain_iters: int = 3     # Actor-Critic-Vortraining (Iterationen)
    initial_ac_pretrain_episodes: int = 4  # Episoden pro Vortrainings-Iteration

    # --- Actor-Critic-Updates ---
    ac_sample_episodes: int = 16           # Startzustaende fuer die Imagination pro Update
    ac_updates_per_iter: int = 10          # mehr Policy-Updates -> schnelleres Actor-Lernen
    clip_grad_norm: float = 100.0          # Gradient-Clipping (DreamerV2 Tab. D.1)

    # --- Datensammlung & Evaluation ---
    collect_episodes_per_iter: int = 6     # frische Umgebungsdaten pro Iteration
    max_steps: int = 500                   # max. Schritte pro Episode (Breakout endet sonst selbst)
    train_interval_episodes: int = 4       # (reserviert, derzeit ungenutzt)
    explore_epsilon: float = 0.0           # Exploration laeuft ueber Actor-Entropie, nicht epsilon
    eval_episodes: int = 10                # Episoden pro Evaluation


def build_actor_critic(world_model):
    feat_size = world_model.stoch_size + world_model.h_size
    actor = Actor(feat_size).to(DEVICE)
    critic = Critic(feat_size).to(DEVICE)
    # Kein ObsActor mehr: der Agent handelt ueber den mitgefuehrten RSSM-Zustand
    # (world_model.act_step) auf denselben Latent-Features [flatz, h], auf denen
    # auch das Imagination-Training laeuft. Das behebt die Repraesentations-
    # Diskrepanz zwischen Training und Ausfuehrung.
    return actor, critic


def iterative_train(cfg=TrainConfig()):
    """Trainiert Weltmodell + Actor + Critic iterativ und gibt die BESTE
    (nicht die letzte) Variante zurueck. Gibt am Ende das Modell mit dem
    hoechsten eval_return zurueck -- schuetzt vor spaetem Policy-Collapse."""
    env = None   # [BREAKOUT] native MinAtar-Umgebung wird in collect/evaluate erstellt
    replay_buffer = ReplayBuffer(cfg.replay_capacity)

    world_model = RSSM().to(DEVICE)
    actor, critic = build_actor_critic(world_model)

    # Pro Lauf zuruecksetzen: Target-Netzwerk (stabile Critic-Targets) und
    # ret_scale (laufende Skala fuer die Critic-Loss-Normalisierung).
    if hasattr(train_actor_critic, "_target_critic"):
        del train_actor_critic._target_critic
    if hasattr(train_actor_critic, "_ret_scale"):
        del train_actor_critic._ret_scale

    # Best-Modell-Tracking zuruecksetzen (fuer diesen Lauf).
    iterative_train._best_return = -1.0
    iterative_train._best_state = None
    eval_history = []   # [BREAKOUT] eval_return pro Iteration fuer die Lernkurve

    wm_opt = optim.Adam(world_model.parameters(), lr=LR_WORLD)
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=1e-6)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=1e-6)

    # Initiale Zufallsdaten (epsilon=1.0). Hier reicht der actor als Platzhalter,
    # da rein zufaellig gehandelt wird; RSSM-Zustand wird trotzdem korrekt gefuehrt.
    for ep in collect_episodes(env, actor, world_model, n_episodes=cfg.initial_random_episodes, max_steps=cfg.max_steps, seed=42, epsilon=1.0):
        replay_buffer.add_episode(ep)

    if cfg.initial_world_model_epochs > 0:
        init_wm_metrics = train_world_model(
            world_model, wm_opt, replay_buffer,
            seq_len=cfg.seq_len, batch_size=cfg.batch_size,
            epochs_per_phase=cfg.initial_world_model_epochs,
        )
        print(f"init_wm buffer={len(replay_buffer):03d} init_wm={init_wm_metrics}")

    for pre_it in range(cfg.initial_ac_pretrain_iters):
        new_eps = collect_episodes(env, actor, world_model, n_episodes=cfg.initial_ac_pretrain_episodes,
                                   max_steps=cfg.max_steps, epsilon=cfg.explore_epsilon)
        for ep in new_eps:
            replay_buffer.add_episode(ep)

        for _ in range(cfg.ac_updates_per_iter):
            pre_ac_metrics = train_actor_critic(
                world_model, actor, critic, actor_opt, critic_opt, replay_buffer,
                imagination_horizon=cfg.imagination_horizon, warmup_steps=cfg.warmup_steps,
                sample_episodes=cfg.ac_sample_episodes, entropy_coeff=cfg.actor_entropy_coeff,
                clip_norm=cfg.clip_grad_norm,
            )
        pre_eval_return = evaluate_policy(env, world_model, actor, episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
        print(f"pretrain={pre_it+1}/{cfg.initial_ac_pretrain_iters} buffer={len(replay_buffer):03d} ac={pre_ac_metrics} eval_return={pre_eval_return:.2f}")

    for it in range(cfg.iterations):
        # Entropie-Annealing: linear von entropy_start (viel Exploration) zu
        # entropy_end (feine Policy) ueber den Trainingsverlauf.
        frac = it / max(1, cfg.iterations - 1)
        ent_coeff = cfg.entropy_start + frac * (cfg.entropy_end - cfg.entropy_start)

        new_eps = collect_episodes(env, actor, world_model, n_episodes=cfg.collect_episodes_per_iter,
                                   max_steps=cfg.max_steps, epsilon=cfg.explore_epsilon)
        for ep in new_eps:
            replay_buffer.add_episode(ep)

        wm_metrics = train_world_model(
            world_model, wm_opt, replay_buffer,
            seq_len=cfg.seq_len, batch_size=cfg.batch_size, epochs_per_phase=cfg.epochs_per_phase,
        )
        for _ in range(cfg.ac_updates_per_iter):
            ac_metrics = train_actor_critic(
                world_model, actor, critic, actor_opt, critic_opt, replay_buffer,
                imagination_horizon=cfg.imagination_horizon, warmup_steps=cfg.warmup_steps,
                sample_episodes=cfg.ac_sample_episodes, entropy_coeff=ent_coeff,
                clip_norm=cfg.clip_grad_norm,
            )

        eval_return = evaluate_policy(env, world_model, actor, episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
        eval_history.append(eval_return)   # [BREAKOUT] fuer die Lernkurve
        print(f"iter={it:03d} buffer={len(replay_buffer):03d} wm={wm_metrics} ac={ac_metrics} eval_return={eval_return:.2f}")

        # Bestes Modell festhalten (gegen Policy-Collapse nach dem Peak):
        # wir behalten die Gewichte mit dem hoechsten eval_return, nicht die letzten.
        if eval_return > iterative_train._best_return:
            iterative_train._best_return = eval_return
            iterative_train._best_state = (
                {k: v.detach().cpu().clone() for k, v in world_model.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
            )

    # [BREAKOUT] kein env.close() noetig -- Umgebungen werden pro Episode erstellt

    # Falls ein besseres Zwischenmodell existiert, dieses zurueckgeben (nicht das letzte).
    if iterative_train._best_state is not None:
        print(f"\n[INFO] Bestes Modell hatte eval_return={iterative_train._best_return:.1f} "
              f"(letztes: {eval_return:.1f}). Lade bestes Modell.")
        wm_state, actor_state, critic_state = iterative_train._best_state
        world_model.load_state_dict(wm_state)
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)

    # [BREAKOUT] Lernkurven-Verlauf speichern (von analysis_breakout genutzt).
    import numpy as _np
    from pathlib import Path as _Path
    _np.save(_Path(__file__).resolve().parent / "training_log_breakout.npy",
             _np.array(eval_history, dtype=_np.float32))

    return world_model, actor, critic


if __name__ == "__main__":
    # Startet das ausführliche Training
    world_model, actor, critic = iterative_train()

    # Speichern für die Analyse-Skripte
    from pathlib import Path
    save_dir = Path(__file__).resolve().parent
    torch.save(world_model.state_dict(), save_dir / "world_model_breakout.pth")
    torch.save(actor.state_dict(), save_dir / "actor_breakout.pth")
    torch.save(critic.state_dict(), save_dir / "critic_breakout.pth")
    print(f"\n[INFO] Modelle erfolgreich in {save_dir} für die Analyse gesichert!")