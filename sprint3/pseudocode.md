═══════════════════════════════════════════════════════════
KONSTANTEN
═══════════════════════════════════════════════════════════
C = 4              # Anzahl kategorischer Variablen
K = 8              # Klassen pro Variable
STOCH_SIZE = C*K   # = 32, flaches stochastisches Latent
DETER_SIZE = 8     # Größe von h (GRU-Memory)
OBS_SIZE = 2       # nur cart_position, pole_angle (keine Velocities!)
ACTION_SIZE = 2


═══════════════════════════════════════════════════════════
DATENERFASSUNG
═══════════════════════════════════════════════════════════
function collect_episodes(env, n_episodes):
    for each episode:
        obs = env.reset()
        while not done:
            a = random_action()
            next_obs, r, done = env.step(a)
            speichere: vis(obs), onehot(a), r, continue_flag
            obs = next_obs
        speichere finalen vis(obs)   # Länge T+1 für Observations
    return episodes

function visible_state(full_state):
    return full_state[[0, 2]]   # nur Position + Winkel


═══════════════════════════════════════════════════════════
SEQUENCE DATASET
═══════════════════════════════════════════════════════════
für jede Episode:
    schneide überlappende Fenster der Länge seq_len heraus
    → (observations[T+1], actions[T], rewards[T], continues[T])


═══════════════════════════════════════════════════════════
RSSM — KERNSCHLEIFE (observe_forward)
═══════════════════════════════════════════════════════════
function observe_forward(observations, actions):
    h, z = 0, 0   # Initialzustand

    for t in 0..T-1:
    
        # 1) Aktion + voriges Latent → Feature
        feat = MLP_action(concat(z, action[t]))

        # 2) Deterministisches Memory updaten
        h = GRU(feat, h)

        # 3) PRIOR: sieht NUR h (keine Zukunft!)
        prior_logits = MLP_prior(h)            # shape (B,C,K)
        prior_z = sample_straight_through(prior_logits)
        prior_pred = predict(prior_z, h)        # obs, reward, continue

        # 4) POSTERIOR: sieht h + reale nächste Beobachtung
        post_logits = MLP_posterior(concat(h, obs[t+1]))
        post_z = sample_straight_through(post_logits)
        post_pred = predict(post_z, h)

        # 5) WICHTIG: nächster Schritt nutzt Posterior-Latent
        z = post_z

        speichere prior_logits, post_logits, prior_pred, post_pred

    return alle gespeicherten Listen (gestapelt über t)


function predict(z, h):
    feat = MLP_pred(concat(flatten(z), h))
    return {
        observation: Linear_obs(feat),
        reward:      Linear_reward(feat),
        continue:    Linear_continue(feat)   # roher Logit
    }


═══════════════════════════════════════════════════════════
STRAIGHT-THROUGH SAMPLING
═══════════════════════════════════════════════════════════
function sample_straight_through(logits):
    probs = softmax(logits)
    onehot = sample_kategorisch(probs)      # forward: hart
    return onehot.detach() - probs.detach() + probs
    #      ^ Vorwärts = onehot, Rückwärts = Gradient von probs


═══════════════════════════════════════════════════════════
PRIOR ROLLOUT (Imagination, ohne Zukunftsdaten)
═══════════════════════════════════════════════════════════
function prior_rollout(observations, actions, warmup_steps):
    h, z = 0, 0

    # Phase 1: Warmup MIT echten Beobachtungen (Posterior, Mode)
    for t in 0..warmup_steps-1:
        h = GRU(MLP_action(z, action[t]), h)
        z = mode(MLP_posterior(h, obs[t+1]))   # argmax statt sample

    # Phase 2: Rollout NUR mit Prior (keine echten Beobachtungen mehr!)
    preds = []
    for t in warmup_steps..T-1:
        h = GRU(MLP_action(z, action[t]), h)
        z = mode(MLP_prior(h))                  # <- kein obs[t+1]!
        preds.append(predict(z, h).observation)

    return preds


═══════════════════════════════════════════════════════════
VERLUSTFUNKTION
═══════════════════════════════════════════════════════════
function compute_losses(model_out, obs, rewards, continues):
    target_obs = obs[1:]   # verschoben um 1, da action[t] → obs[t+1]

    # Posterior-Verluste (Hauptsignal)
    L_post = MSE(post_pred.obs, target_obs)
           + reward_scale * MSE(post_pred.reward, rewards)
           + cont_scale   * BCE(post_pred.continue_logit, continues)

    # Prior-Verluste (Prior soll auch gut vorhersagen können)
    L_prior = MSE(prior_pred.obs, target_obs)
            + reward_scale * MSE(prior_pred.reward, rewards)
            + cont_scale   * BCE(prior_pred.continue_logit, continues)

    # KL zwischen Posterior und Prior (über K, dann über C, dann Mittel über B,T)
    KL = sum_K[ q * (log q - log p) ]  →  sum_C  →  mean_{B,T}

    total_loss = L_post + prior_scale * L_prior + kl_scale * KL
    return total_loss


═══════════════════════════════════════════════════════════
TRAININGSSCHLEIFE
═══════════════════════════════════════════════════════════
for epoch in 1..N:
    for batch in dataloader:
        out  = observe_forward(batch.obs, batch.actions)
        loss = compute_losses(out, batch.obs, batch.rewards, batch.continues)
        loss.backward()
        optimizer.step()
    print(epoch_metrics)
    test prior_rollout(sample)   # Sanity-Check pro Epoche