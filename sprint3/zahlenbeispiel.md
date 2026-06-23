═══════════════════════════════════════════════════════════
SCHRITT 0: Initialzustand
═══════════════════════════════════════════════════════════
h = [0, 0, 0, 0]              # noch kein Gedächtnis vorhanden
z = [0,0,0, 0,0,0]             # noch kein Latent-Zustand gewählt


═══════════════════════════════════════════════════════════
EINGABEDATEN für Zeitschritt t=0
═══════════════════════════════════════════════════════════
obs[0] = [0.10, 0.05]          # echte Beobachtung jetzt (Position, Winkel)
obs[1] = [0.12, 0.09]          # echte Beobachtung einen Schritt später -- DAS will das Modell vorhersagen
action[0] = [1, 0]             # one-hot: Aktion "links"
reward[0] = [1.0]              # CartPole gibt +1 pro überlebtem Schritt
continue[0] = [1.0]            # Episode läuft nach diesem Schritt noch weiter


═══════════════════════════════════════════════════════════
SCHRITT 1 (RECURRENT MODEL): h aktualisieren
═══════════════════════════════════════════════════════════
feat = MLP_action(concat(z, action[0]))
     = MLP_action([0,0,0,0,0,0, 1,0])          # z und Aktion zusammengehängt (8 Werte)
     = [0.3, -0.1, 0.5, 0.2]                   # komprimiert auf 4 Werte:
                                                # "was bedeutet diese Aktion gegeben
                                                #  den (hier leeren) vorigen Zustand?"

h = GRU(feat, h_alt)
  = GRU([0.3,-0.1,0.5,0.2], [0,0,0,0])
  = [0.21, -0.05, 0.33, 0.14]                  # neues Gedächtnis: GRU entscheidet,
                                                # wie viel vom alten h behalten wird
                                                # (hier: alles 0, also komplett neu)


═══════════════════════════════════════════════════════════
SCHRITT 2 (PRIOR MODEL): blind raten, nur aus h
═══════════════════════════════════════════════════════════
prior_logits_flat = MLP_prior(h)
                   = [2.1, 0.3, -1.0,   0.5, 1.8, -0.2]   # rohe Scores, noch keine Wahrscheinlichkeiten

prior_logits = [[2.1, 0.3, -1.0],    # Kategorie 1: 3 Scores
                [0.5, 1.8, -0.2]]    # Kategorie 2: 3 Scores

probs_kat1 = softmax([2.1, 0.3, -1.0]) = [0.74, 0.18, 0.08]
probs_kat2 = softmax([0.5, 1.8, -0.2]) = [0.22, 0.70, 0.08]
            # Scores -> echte Wahrscheinlichkeiten (summieren sich zu 1)
            # Prior ist hier noch recht unsicher (höchster Wert nur 0.74),
            # weil er die echte Zukunft NICHT sehen darf

prior_z = [[1,0,0],     # gewürfelt: Klasse 0 gewonnen (war auch am wahrscheinlichsten)
           [0,1,0]]     # gewürfelt: Klasse 1 gewonnen

prior_z_flat = [1,0,0, 0,1,0]   # für die nächsten Linear-Schichten flach gemacht


═══════════════════════════════════════════════════════════
SCHRITT 3 (POSTERIOR MODEL): raten MIT Schummelzettel
═══════════════════════════════════════════════════════════
posterior_logits_flat = MLP_posterior(concat(h, obs[1]))
                       = [3.5, 0.1, -2.0,   0.1, 2.5, -0.5]
                       # Unterschied zum Prior: bekommt zusätzlich obs[1] zu sehen!

posterior_logits = [[3.5, 0.1, -2.0],
                     [0.1, 2.5, -0.5]]

probs_kat1 = softmax([3.5, 0.1, -2.0]) = [0.95, 0.04, 0.01]
probs_kat2 = softmax([0.1, 2.5, -0.5]) = [0.09, 0.86, 0.05]
            # Deutlich sicherer als der Prior (0.95 statt 0.74)!
            # Macht Sinn: wer die Antwort schon kennt, kann sicherer raten

post_z = [[1,0,0],    # zufällig dieselbe Klasse wie Prior in diesem Beispiel
          [0,1,0]]    # (muss nicht immer so sein)

post_z_flat = [1,0,0, 0,1,0]


═══════════════════════════════════════════════════════════
SCHRITT 4: Vorhersagen aus beiden Latents ableiten
═══════════════════════════════════════════════════════════
feat_prior = MLP_pred(concat(prior_z_flat, h))
pred_prior.observation = [0.11, 0.07]    # Modell schätzt: nächste obs wird ~[0.11,0.07]
pred_prior.reward      = [0.95]          # Modell schätzt: Reward wird ~0.95
pred_prior.continue    = [2.1]           # roher Logit -> sigmoid(2.1)≈0.89 "läuft weiter"

feat_post = MLP_pred(concat(post_z_flat, h))
pred_post.observation = [0.13, 0.08]     # Posterior-Vorhersage, etwas näher an Wahrheit
pred_post.reward      = [0.98]
pred_post.continue    = [2.4]            # sigmoid(2.4)≈0.92


═══════════════════════════════════════════════════════════
SCHRITT 5: Verluste berechnen -- wie falsch lagen wir?
═══════════════════════════════════════════════════════════
target_obs = obs[1] = [0.12, 0.09]       # das ist die Wahrheit

obs_mse_post  = MSE([0.13,0.08], [0.12,0.09]) ≈ 0.0001
              # quadrierter Fehler, sehr klein -> Vorhersage war fast perfekt

rew_mse_post  = MSE([0.98], [1.0]) ≈ 0.0004
              # auch sehr klein -> Reward-Vorhersage fast perfekt

cont_bce_post = BCE(logit=2.4, target=1.0) ≈ 0.086
              # sigmoid(2.4)=0.917, Fehler = -log(0.917) -> größter Verlustanteil hier,
              # weil die Continue-Vorhersage am "unsichersten" war

obs_mse_prior = MSE([0.11,0.07], [0.12,0.09]) ≈ 0.0002
rew_mse_prior = MSE([0.95], [1.0]) ≈ 0.0025
cont_bce_prior= BCE(logit=2.1, target=1.0) ≈ 0.115
              # Prior-Fehler durchgehend etwas größer als Posterior-Fehler,
              # weil Prior ohne Schummelzettel raten musste


─── KL-Divergenz: wie unterschiedlich sind Prior und Posterior? ───
KL_kat1 = 0.95*log(0.95/0.74) + 0.04*log(0.04/0.18) + 0.01*log(0.01/0.08)
        ≈ 0.95*0.249 + 0.04*(-1.504) + 0.01*(-2.079)
        ≈ 0.237 - 0.060 - 0.021 ≈ 0.156
        # Erster Term positiv: Posterior glaubt stärker an Klasse 0 als Prior
        # -> Prior "verschwendet" hier Wahrscheinlichkeit, wird bestraft

KL_kat2 ≈ 0.09*(-0.875) + 0.86*0.207 + 0.05*0.182 ≈ 0.108

KL_total_t0 = KL_kat1 + KL_kat2 ≈ 0.264
            # Je größer dieser Wert, desto unterschiedlicher sind Prior und Posterior.
            # Der Trainings-Gradient zieht den Prior näher an den Posterior heran.


════════════════════════════════════════════════