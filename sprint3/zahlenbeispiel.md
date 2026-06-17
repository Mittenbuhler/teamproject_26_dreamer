Schritt 0: Initialzustand
h = [0, 0, 0, 0]                          # DETER_SIZE=4
z = [0,0,0, 0,0,0]                        # STOCH_SIZE=6 (2 Kategorien × 3 Klassen, flach)

Eingabedaten für Zeitschritt t=0
Nehmen wir eine konkrete Beobachtung und Aktion:
obs[0] = [0.10, 0.05]      # Position=0.10, Winkel=0.05 (echte Beobachtung zum Zeitpunkt 0)
obs[1] = [0.12, 0.09]      # echte Beobachtung zum Zeitpunkt 1 (das Ziel für die Vorhersage)
action[0] = [1, 0]         # one-hot: Aktion "links"
reward[0] = [1.0]
continue[0] = [1.0]        # Episode läuft noch


Schritt 1: Recurrent Model — h aktualisieren
feat = MLP_action(concat(z, action[0]))
     = MLP_action([0,0,0,0,0,0, 1,0])     # 8 Werte rein
     = [0.3, -0.1, 0.5, 0.2]              # 4 Werte raus (Beispielzahlen)

h = GRU(feat, h_alt)
  = GRU([0.3,-0.1,0.5,0.2], [0,0,0,0])
  = [0.21, -0.05, 0.33, 0.14]             # neuer h_0


Schritt 2: Prior Model — blinde Vorhersage von z
prior_logits_flat = MLP_prior(h)
                   = MLP_prior([0.21, -0.05, 0.33, 0.14])
                   = [2.1, 0.3, -1.0,   0.5, 1.8, -0.2]   # 6 Werte (C*K)

prior_logits = reshape → [[2.1, 0.3, -1.0],     # Kategorie 1: Logits für 3 Klassen
                           [0.5, 1.8, -0.2]]     # Kategorie 2: Logits für 3 Klassen

# Softmax pro Kategorie:
probs_kat1 = softmax([2.1, 0.3, -1.0]) = [0.74, 0.18, 0.08]
probs_kat2 = softmax([0.5, 1.8, -0.2]) = [0.22, 0.70, 0.08]

# Sample (forward = hart):
prior_z = [[1,0,0],     # Kategorie 1 hat Klasse 0 gewählt (höchste Wahrscheinlichkeit)
           [0,1,0]]     # Kategorie 2 hat Klasse 1 gewählt

flatten → prior_z_flat = [1,0,0, 0,1,0]


Schritt 3: Posterior Model — Vorhersage mit Schummelzettel (echte obs[1])
posterior_logits_flat = MLP_posterior(concat(h, obs[1]))
                       = MLP_posterior([0.21,-0.05,0.33,0.14, 0.12,0.09])  # 6 Werte rein
                       = [3.5, 0.1, -2.0,   0.1, 2.5, -0.5]                # 6 Werte raus

posterior_logits = [[3.5, 0.1, -2.0],
                     [0.1, 2.5, -0.5]]

probs_kat1 = softmax([3.5, 0.1, -2.0]) = [0.95, 0.04, 0.01]
probs_kat2 = softmax([0.1, 2.5, -0.5]) = [0.09, 0.86, 0.05]

post_z = [[1,0,0],    # auch hier Klasse 0
          [0,1,0]]    # auch hier Klasse 1

post_z_flat = [1,0,0, 0,1,0]
Beachte: Posterior ist hier sehr sicher (0.95, 0.86), weil es ja den "Schummelzettel" (echtes obs[1]) gesehen hat. Prior ist unsicherer (0.74, 0.70), weil es blind raten musste — genau das erwarten wir am Anfang des Trainings.


Schritt 4: Vorhersagen aus beiden Latents
# Aus prior_z_flat:
feat_prior = MLP_pred(concat([1,0,0,0,1,0], [0.21,-0.05,0.33,0.14]))
pred_prior.observation = Linear_obs(feat_prior) = [0.11, 0.07]   # Vorhersage für obs[1]
pred_prior.reward      = Linear_reward(feat_prior) = [0.95]
pred_prior.continue    = Linear_continue(feat_prior) = [2.1]     # roher Logit, sigmoid(2.1)≈0.89

# Aus post_z_flat (zufällig identisch in diesem Beispiel, da gleiche Klassen gewählt):
feat_post = MLP_pred(concat([1,0,0,0,1,0], [0.21,-0.05,0.33,0.14]))
pred_post.observation = [0.13, 0.08]    # etwas näher am echten obs[1]=[0.12,0.09]
pred_post.reward      = [0.98]
pred_post.continue    = [2.4]           # sigmoid(2.4)≈0.92


Schritt 5: Verluste für diesen einen Zeitschritt
target_obs = obs[1] = [0.12, 0.09]

# Posterior-Verlust:
obs_mse_post  = MSE([0.13,0.08], [0.12,0.09]) ≈ 0.0001
rew_mse_post  = MSE([0.98], [1.0]) ≈ 0.0004
cont_bce_post = BCE(logit=2.4, target=1.0) ≈ 0.086

# Prior-Verlust:
obs_mse_prior = MSE([0.11,0.07], [0.12,0.09]) ≈ 0.0002
rew_mse_prior = MSE([0.95], [1.0]) ≈ 0.0025
cont_bce_prior= BCE(logit=2.1, target=1.0) ≈ 0.115

# KL(Posterior || Prior), pro Kategorie:
KL_kat1 = sum( post_probs * (log(post_probs) - log(prior_probs)) )
        = 0.95*log(0.95/0.74) + 0.04*log(0.04/0.18) + 0.01*log(0.01/0.08)
        ≈ 0.95*0.249 + 0.04*(-1.504) + 0.01*(-2.079)
        ≈ 0.237 - 0.060 - 0.021 ≈ 0.156

KL_kat2 = sum( post_probs * (log(post_probs) - log(prior_probs)) )
        ≈ 0.09*(-0.875) + 0.86*0.207 + 0.05*0.182
        ≈ -0.079 + 0.178 + 0.009 ≈ 0.108

KL_total_t0 = KL_kat1 + KL_kat2 ≈ 0.156 + 0.108 = 0.264


Schritt 6: Gesamtverlust für diesen Zeitschritt (mit deinen echten Skalierungsfaktoren)
L_post  = 0.0001 + 2.0*0.0004 + 1.0*0.086 ≈ 0.087
L_prior = 0.0002 + 2.0*0.0025 + 1.0*0.115 ≈ 0.120

total_loss_t0 = L_post + 0.5*L_prior + 0.1*KL_total_t0
              = 0.087 + 0.5*0.120 + 0.1*0.264
              = 0.087 + 0.060 + 0.026
              = 0.174
