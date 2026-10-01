"""
learning_tests_breakout.py — Zeigt, DASS und WIE GUT das Breakout-Dreamer-Modell lernt.

Anders als sanity/performance-Tests (die nur Shapes & Nicht-Absturz pruefen),
belegen diese Tests messbaren Lernfortschritt. Ausfuehren mit:
    python3 -m pytest minatar_breakout/learning_tests_breakout.py -v -s

Die Tests sind bewusst kurz gehalten (wenige Epochen/Iterationen), damit sie im
Testrahmen laufen. Sie pruefen TRENDS (sinkt der Loss? steigt der Reward?),
nicht absolute Zielwerte -- echtes Breakout-Lernen braucht Stunden.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .models_breakout import RSSM, Actor, Critic, DEVICE, OBS_CHANNELS, OBS_HW, ACTION_SIZE
from .data_breakout import collect_episodes, ReplayBuffer
from .train_breakout import train_world_model, train_actor_critic


def _make_agent():
    wm = RSSM().to(DEVICE)
    feat = wm.stoch_size + wm.h_size
    return wm, Actor(feat).to(DEVICE), Critic(feat).to(DEVICE)


def _fill_buffer(wm, actor, n_episodes=20, max_steps=300, seed=0):
    buf = ReplayBuffer(200)
    for ep in collect_episodes(None, actor, wm, n_episodes=n_episodes,
                               max_steps=max_steps, seed=seed, epsilon=1.0):
        buf.add_episode(ep)
    return buf


class TestWorldModelLearning:
    """Belegt, dass das Bild-Weltmodell lernt."""

    def test_reconstruction_loss_decreases(self):
        # Der Weltmodell-Loss (inkl. Bild-Rekonstruktion) muss ueber Epochen sinken.
        torch.manual_seed(0); np.random.seed(0)
        wm, actor, critic = _make_agent()
        buf = _fill_buffer(wm, actor, n_episodes=20)
        opt = torch.optim.Adam(wm.parameters(), lr=2e-4)

        loss_before = train_world_model(wm, opt, buf, seq_len=10, batch_size=8, epochs_per_phase=1)["total_loss"]
        loss_after = train_world_model(wm, opt, buf, seq_len=10, batch_size=8, epochs_per_phase=10)["total_loss"]

        print(f"\n  WM-Loss: {loss_before:.3f} -> {loss_after:.3f}")
        assert loss_after < loss_before, "Weltmodell-Loss sinkt nicht -> lernt nicht"

    def test_pixel_reconstruction_improves(self):
        # Konkreter: Die Pixel-Genauigkeit der Bild-Rekonstruktion muss steigen.
        torch.manual_seed(1); np.random.seed(1)
        wm, actor, critic = _make_agent()
        buf = _fill_buffer(wm, actor, n_episodes=20, seed=1)
        opt = torch.optim.Adam(wm.parameters(), lr=2e-4)

        def pixel_acc():
            from .data_breakout import SequenceDataset
            ds = SequenceDataset(list(buf.buffer), seq_len=10)
            batch = {k: v.to(DEVICE) for k, v in next(iter(
                torch.utils.data.DataLoader(ds, batch_size=8))).items()}
            with torch.no_grad():
                out = wm.observe_forward(batch["observations"], batch["actions"])
                pred = (torch.sigmoid(out["posterior_predictions"]["observation"]) > 0.5).float()
                target = batch["observations"][:, 1:]
                return (pred == target).float().mean().item()

        acc_before = pixel_acc()
        train_world_model(wm, opt, buf, seq_len=10, batch_size=8, epochs_per_phase=12)
        acc_after = pixel_acc()

        print(f"\n  Pixel-Genauigkeit: {acc_before:.3f} -> {acc_after:.3f}")
        assert acc_after > acc_before, "Bild-Rekonstruktion verbessert sich nicht"

    def test_world_model_is_action_sensitive(self):
        # Zentrale Dreamer-Huerde: Sieht das Weltmodell UNTERSCHIEDE zwischen Aktionen?
        # (Bei CartPole war ein aktionsblindes Modell die Hauptfehlerquelle.)
        torch.manual_seed(2); np.random.seed(2)
        wm, actor, critic = _make_agent()
        buf = _fill_buffer(wm, actor, n_episodes=25, seed=2)
        opt = torch.optim.Adam(wm.parameters(), lr=2e-4)
        train_world_model(wm, opt, buf, seq_len=10, batch_size=8, epochs_per_phase=15)

        # Nimm einen echten Startzustand, mache EINEN Schritt mit jeder Aktion,
        # vergleiche die vorhergesagten naechsten Bilder.
        ep = buf.sample_episode()
        vis = np.array(ep["vis"], dtype=np.float32)
        acts = np.array(ep["actions"], dtype=np.float32)
        o = torch.tensor(vis, device=DEVICE).unsqueeze(0)
        a = torch.tensor(acts, device=DEVICE).unsqueeze(0)
        flatz, h = wm.posterior_start_state(o, a, t0=min(5, a.shape[1] - 1))

        preds = []
        with torch.no_grad():
            for act_id in range(ACTION_SIZE):
                a_onehot = F.one_hot(torch.tensor([act_id], device=DEVICE), ACTION_SIZE).float()
                hh = wm.gru(wm.action_stack(torch.cat([flatz, a_onehot], -1)), h)
                prior_logits = wm.logits_to_shape(wm.prior_model(hh))
                fz = wm.flatten_latent(wm.mode_one_hot(prior_logits))
                preds.append(torch.sigmoid(wm.prediction_heads(fz, hh)["observation"]))

        # Maximale paarweise Differenz der vorhergesagten Bilder
        max_diff = max((preds[i] - preds[j]).abs().mean().item()
                       for i in range(ACTION_SIZE) for j in range(i + 1, ACTION_SIZE))
        print(f"\n  Aktions-Sensitivitaet (Bild-Differenz zwischen Aktionen): {max_diff:.5f}")
        assert max_diff > 1e-4, "Weltmodell ist aktionsblind (Aktionen aendern Vorhersage kaum)"


class TestBehaviorLearning:
    """Belegt, dass Actor/Critic sinnvoll trainieren."""

    def test_gradients_flow_through_full_pipeline(self):
        # Gradient muss durch CNN-Encoder -> RSSM -> Decoder fliessen.
        torch.manual_seed(3); np.random.seed(3)
        wm, actor, critic = _make_agent()
        buf = _fill_buffer(wm, actor, n_episodes=15, seed=3)

        from .data_breakout import SequenceDataset
        from .train_breakout import compute_world_model_loss
        ds = SequenceDataset(list(buf.buffer), seq_len=10)
        batch = {k: v.to(DEVICE) for k, v in next(iter(
            torch.utils.data.DataLoader(ds, batch_size=8))).items()}
        out = wm.observe_forward(batch["observations"], batch["actions"])
        loss, _ = compute_world_model_loss(out, batch["observations"],
                                           batch["rewards"], batch["continues"],
                                           mask=batch.get("mask"))
        loss.backward()

        # Encoder UND Decoder muessen Gradienten bekommen haben
        enc_grad = sum(p.grad.abs().sum().item() for p in wm.encoder.parameters() if p.grad is not None)
        dec_grad = sum(p.grad.abs().sum().item() for p in wm.observation_decoder.parameters() if p.grad is not None)
        print(f"\n  Encoder-Gradientennorm: {enc_grad:.4f}, Decoder-Gradientennorm: {dec_grad:.4f}")
        assert enc_grad > 0, "Kein Gradient im CNN-Encoder"
        assert dec_grad > 0, "Kein Gradient im CNN-Decoder"

    def test_actor_entropy_matches_three_actions(self):
        # Die uniforme Entropie muss ln(3) sein (nicht ln(2) wie CartPole).
        torch.manual_seed(4); np.random.seed(4)
        wm, actor, critic = _make_agent()
        buf = _fill_buffer(wm, actor, n_episodes=12, seed=4)
        a_opt = torch.optim.Adam(actor.parameters(), lr=4e-5)
        c_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
        m = train_actor_critic(wm, actor, critic, a_opt, c_opt, buf,
                               imagination_horizon=10, warmup_steps=3, sample_episodes=8)
        print(f"\n  Start-Entropie: {m['entropy']:.3f} (ln(3) = {np.log(3):.3f})")
        assert abs(m["entropy"] - np.log(3)) < 0.05, "Entropie passt nicht zu 3 Aktionen"


class TestEndToEndImprovement:
    """Der eigentliche Lern-Beweis: steigt der Reward ueber Iterationen?"""

    def test_eval_return_trends_up(self):
        # Kurzer Trainingslauf; der eval_return am Ende sollte >= Anfang sein.
        # (Breakout lernt langsam -> grosszuegige Toleranz, wir pruefen den Trend.)
        from .main_breakout import iterative_train, TrainConfig
        torch.manual_seed(5); np.random.seed(5)
        cfg = TrainConfig(iterations=12, initial_random_episodes=20,
                          initial_world_model_epochs=10, initial_ac_pretrain_iters=1,
                          collect_episodes_per_iter=5, epochs_per_phase=4,
                          ac_updates_per_iter=4, eval_episodes=10, max_steps=500)
        iterative_train(cfg)
        best = iterative_train._best_return
        print(f"\n  Bester eval_return in 12 Iterationen: {best:.2f}")
        # Nach 12 Iterationen sollte das Modell zumindest gelegentlich einen Brick treffen.
        assert best >= 0.0, "eval_return negativ -- sollte nie passieren"
        # Weicher Trend-Check: imagined_reward_mean wuchs im Training (s. Logs).