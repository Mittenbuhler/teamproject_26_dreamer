import gymnasium as gym
import torch
import torch.optim as optim
from dataclasses import dataclass
from .models import RSSM, Actor, Critic, DEVICE
from .data import ReplayBuffer, collect_episodes, visible_state, onehot_action
from .train import train_world_model, train_actor_critic

"""ausführen mit command: python -m sprint5.main"""


def evaluate_policy(env, world_model, actor, episodes=5, max_steps=200):
    """Evaluiert die Policy ueber den kanonischen RSSM-Handel: der RSSM-Zustand
    wird ueber die Episode mitgefuehrt, die Policy handelt auf [flatz, h]."""
    world_model.eval()
    actor.eval()
    returns = []
    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=ep_idx)
        total_reward = 0.0
        done = False
        steps = 0

        flatz, h = world_model.initial(batch_size=1, device=DEVICE)
        prev_action = torch.zeros(1, world_model.action_size, device=DEVICE)
        obs_t = torch.tensor(visible_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)

        while not done and steps < max_steps:
            with torch.no_grad():
                logits = actor(feat)
                action = torch.distributions.Categorical(logits=logits).sample().item()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += r
            steps += 1

            prev_action = torch.tensor(onehot_action(action), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            obs_t = torch.tensor(visible_state(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flatz, h, feat = world_model.act_step(flatz, h, prev_action, obs_t)
        returns.append(total_reward)
    return sum(returns) / len(returns)


LR_WORLD = 2e-4   # DreamerV2 world model lr (Tab. D.1)
LR_ACTOR = 4e-5   # DreamerV2 actor lr
LR_CRITIC = 1e-4  # DreamerV2 critic lr


@dataclass
class TrainConfig:
    iterations: int = 200                  # laenger trainieren fuer ~200 Reward
    epochs_per_phase: int = 4
    batch_size: int = 32
    seq_len: int = 25
    replay_capacity: int = 2000
    imagination_horizon: int = 15
    actor_entropy_coeff: float = 3e-3      # hoeher: haelt Exploration laenger am Leben (gegen zu fruehen Entropie-Kollaps)
    warmup_steps: int = 5
    initial_random_episodes: int = 15
    initial_world_model_epochs: int = 30
    initial_ac_pretrain_iters: int = 3
    initial_ac_pretrain_episodes: int = 4
    ac_sample_episodes: int = 16
    ac_updates_per_iter: int = 10          # mehr Policy-Updates -> schnelleres Actor-Lernen
    clip_grad_norm: float = 100.0          # DreamerV2 gradient clipping (Tab. D.1)
    collect_episodes_per_iter: int = 6     # mehr frische Umgebungsdaten pro Iteration
    max_steps: int = 500
    train_interval_episodes: int = 4
    explore_epsilon: float = 0.0           # Exploration ueber Actor-Entropie, nicht epsilon
    eval_episodes: int = 10


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
    env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(cfg.replay_capacity)

    world_model = RSSM().to(DEVICE)
    actor, critic = build_actor_critic(world_model)

    # frisches Target-Netzwerk pro Trainingslauf
    if hasattr(train_actor_critic, "_target_critic"):
        del train_actor_critic._target_critic

    # Best-Modell-Tracking zuruecksetzen (fuer diesen Lauf).
    iterative_train._best_return = -1.0
    iterative_train._best_state = None

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
                sample_episodes=cfg.ac_sample_episodes, entropy_coeff=cfg.actor_entropy_coeff,
                clip_norm=cfg.clip_grad_norm,
            )

        eval_return = evaluate_policy(env, world_model, actor, episodes=cfg.eval_episodes, max_steps=cfg.max_steps)
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

    env.close()

    # Falls ein besseres Zwischenmodell existiert, dieses zurueckgeben (nicht das letzte).
    if iterative_train._best_state is not None:
        print(f"\n[INFO] Bestes Modell hatte eval_return={iterative_train._best_return:.1f} "
              f"(letztes: {eval_return:.1f}). Lade bestes Modell.")
        wm_state, actor_state, critic_state = iterative_train._best_state
        world_model.load_state_dict(wm_state)
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
    return world_model, actor, critic


if __name__ == "__main__":
    # Startet das ausführliche Training
    world_model, actor, critic = iterative_train()

    # Speichern für die Analyse-Skripte
    from pathlib import Path
    save_dir = Path(__file__).resolve().parent
    torch.save(world_model.state_dict(), save_dir / "world_model.pth")
    torch.save(actor.state_dict(), save_dir / "actor.pth")
    torch.save(critic.state_dict(), save_dir / "critic.pth")
    print(f"\n[INFO] Modelle erfolgreich in {save_dir} für die Analyse gesichert!")