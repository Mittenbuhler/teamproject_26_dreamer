# CartPole – DreamerV2 mit RSSM-Weltmodell

Dieser Ordner (`cartpole/`) enthält unsere Implementierung eines **DreamerV2-Agenten** für die Gymnasium-Umgebung `CartPole-v1`. Der Agent lernt ein Weltmodell (RSSM) aus gesammelten Erfahrungen und trainiert Actor und Critic anschließend ausschließlich in "geträumten" Trajektorien im latenten Raum.

Das Modell kann wahlweise mit 4D-Beobachtungen (vollständiger Zustand) oder 2D-Beobachtungen (nur Position und Winkel, d. h. teilweise beobachtbar) trainiert werden.


---

## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Ordnerstruktur](#ordnerstruktur)
3. [Installation](#installation)
4. [Schnellstart](#schnellstart)
5. [Funktionsweise](#funktionsweise)
6. [Dateien im Detail](#dateien-im-detail)
7. [Hyperparameter](#hyperparameter)
8. [Analyse-Skripte](#analyse-skripte)
9. [Tests](#tests)
10. [Referenzen](#referenzen)

---

## Überblick

| Aspekt | Beschreibung |
|---|---|
| Umgebung | `CartPole-v1` (Gymnasium), max. 500 Schritte pro Episode |
| Algorithmus | DreamerV2 (diskrete Latents, REINFORCE-Actor, λ-Returns) |
| Beobachtungen | 4D: `[x, ẋ, θ, θ̇]` oder 2D: `[x, θ]` |
| Aktionen | 2 diskrete Aktionen (links / rechts), One-Hot kodiert |
| Framework | PyTorch |
| Sprache im Code | Kommentare und Docstrings auf Deutsch |

**Grundidee:**

1. Zufällige Episoden sammeln → Replay Buffer.
2. Weltmodell (RSSM) auf Sequenzen aus dem Buffer trainieren.
3. Actor/Critic in der Imagination des Weltmodells trainieren.
4. Mit dem Actor neue Episoden in der echten Umgebung sammeln → zurück zu 2.

---

## Ordnerstruktur

```
cartpole/
├── __init__.py                # (nötig für `python -m cartpole.<modul>`)
├── models.py                  # RSSM, Actor, Critic (+ ObsActor, ungenutzt)
├── data.py                    # ReplayBuffer, SequenceDataset, Episoden sammeln
├── train.py                   # Weltmodell-Loss, λ-Return, Actor-Critic-Training
├── main.py                    # Trainingsschleife, Evaluation, Speichern der Gewichte
├── make_gif.py                # GIF des balancierenden Agenten
├── analysis_state_model.py    # Analyse: Genauigkeit, Drift, Balancedauer, Policy
├── latent_analysis.py         # Analyse: Probing des Latentraums, PCA, Rekonstruktion
├── sanity_tests.py            # Schnelle Unit-/Shape-Tests
├── performance_tests.py       # Lern- und Laufzeittests
├── models_saved/              # (wird erzeugt) trainierte Gewichte (.pth)
└── plots/                     # (wird erzeugt) Plots und GIFs
```

---

## Installation

Empfohlen wird Python 3.9+. Die benötigten Pakete stehen in der `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Schnellstart

> **Wichtig:** Alle Befehle werden aus dem Ordner **über** `cartpole/` ausgeführt (Projekt-Root), da relative Importe (`from .models import ...`) verwendet werden. Deshalb immer mit `python -m cartpole.<modul>` starten, nicht `python main.py`.

### 1. Trainieren

```bash
python3 -m cartpole.main --inputs 4     # 4D-Modell
python3 -m cartpole.main --inputs 2     # 2D-Modell
```

Die Gewichte landen in `cartpole/models_saved/`:

```
world_model_4d.pth   actor_4d.pth   critic_4d.pth
world_model_2d.pth   actor_2d.pth   critic_2d.pth
```

Während des Trainings wird pro Iteration der `eval_return` ausgegeben. Gespeichert wird das **beste** Modell (höchster `eval_return`), nicht das letzte.

### 2. GIF erzeugen

```bash
python -m cartpole.make_gif --inputs 4
python -m cartpole.make_gif --inputs 4 --episodes 3 --output mein_gif.gif
```

Standardausgabe: `cartpole/plots/cartpole_balance_<2|4>d.gif`

### 3. Analysen ausführen

```bash
python -m cartpole.analysis_state_model --inputs 4
python -m cartpole.latent_analysis --inputs 4
```

Beide benötigen zuvor trainierte Gewichte (Schritt 1).

### 4. Tests

```bash
python -m pytest cartpole/sanity_tests.py -v
python -m pytest cartpole/performance_tests.py -v
```

---

## Funktionsweise

### Weltmodell (RSSM)

Das **Recurrent State Space Model** besteht aus:

- **Deterministischem Zustand `h`** (GRU, Größe 64)
- **Stochastischem Zustand `z`** (kategorisch: 4 Variablen × 8 Klassen = 32 Dimensionen, Straight-Through-Gradienten)
- **Prior** `p(z | h)`: Vorhersage ohne Beobachtung (wird in der Imagination verwendet)
- **Posterior** `q(z | h, obs)`: Schätzung mit echter Beobachtung
- **Prediction Heads**: Beobachtung (2D/4D), Reward und Continue-Signal

Der Feature-Vektor für Actor und Critic ist `feat = [z, h]` (Größe 32 + 64 = 96).

### Training in der Imagination

1. Aus zufälligen Episoden des Replay Buffers wird über Posterior-Updates ein Startzustand aufgebaut (`posterior_start_state`, `warmup_steps`).
2. Ab diesem Zustand "träumt" das Modell `imagination_horizon` Schritte mit der aktuellen Policy (`RSSM.imagine`).
3. Aus den imaginierten Rewards und Critic-Werten werden **λ-Returns** berechnet (mit langsamem Target-Critic).
4. **Critic**: MSE gegen die λ-Returns (discount-gewichtet, skaliert über EMA der Return-Standardabweichung).
5. **Actor**: REINFORCE mit Baseline (Advantage = λ-Return − Critic-Wert, normalisiert) plus Entropie-Bonus.

### Handeln in der echten Umgebung

Der Agent führt seinen RSSM-Zustand über die gesamte Episode mit (`RSSM.act_step`) und wählt Aktionen auf den Features `[z, h]`, also **auf derselben Repräsentation wie im Imagination-Training**. Das gilt für Datensammlung (`data._act_in_env`), Evaluation (`main.evaluate_policy`) und das GIF (`make_gif.py`).

---

## Dateien im Detail

### `models.py`

| Klasse / Konstante | Beschreibung |
|---|---|
| `RSSM(obs_size=4)` | Weltmodell; `obs_size` ∈ {2, 4} bestimmt Posterior-Eingabe und Observation-Head |
| `RSSM.observe_forward` | Forward-Pass über eine Sequenz im Posterior-Modus (für Weltmodell-Training) |
| `RSSM.posterior_start_state` | Startzustand für die Imagination (Argmax-Posterior) |
| `RSSM.act_step` | Ein Schritt in der echten Umgebung, gibt `(flatz, h, feat)` zurück |
| `RSSM.imagine` | Imaginierte Trajektorien für das Actor-Critic-Training |
| `Actor`, `Critic` | MLPs (2 × 128 Hidden), Eingabe: `feat` |
| `ObsActor` | Altes Beobachtungs-basiertes Actor-Modell, **wird nicht mehr verwendet** |

### `data.py`

| Element | Beschreibung |
|---|---|
| `visible_state(state, obs_size)` | Wählt die sichtbaren Zustandswerte (2D: Indizes 0, 2; 4D: alle) |
| `onehot_action(a)` | Aktion → One-Hot-Vektor |
| `ReplayBuffer` | Speichert ganze Episoden (`deque`, `capacity`) |
| `SequenceDataset` | Zerlegt Episoden in überlappungsfreie Fenster der Länge `seq_len`; kürzere Episoden werden mit Null gepaddet und über `mask` maskiert |
| `collect_episodes` | Sammelt Episoden mit der Policy (mit `world_model`, kanonischer Pfad) |

Eine Episode ist ein Dictionary mit den Schlüsseln `fulls`, `vis`, `actions`, `rewards`, `continues`.

### `train.py`

| Funktion | Beschreibung |
|---|---|
| `compute_world_model_loss` | Rekonstruktions-Loss (Posterior + Prior), Reward, Continue, KL-Balancing |
| `lambda_return` | λ-Return nach DreamerV2 Gl. 4 |
| `train_world_model` | Trainiert das RSSM auf Sequenzen aus dem Buffer |
| `train_actor_critic` | Ein Actor-Critic-Update in der Imagination |

### `main.py`

| Element | Beschreibung |
|---|---|
| `TrainConfig` | Dataclass mit allen Trainings-Hyperparametern |
| `iterative_train` | Komplette Trainingsschleife, gibt `(world_model, actor, critic)` zurück |
| `evaluate_policy` | Durchschnittlicher Return über mehrere Episoden |
| `build_actor_critic` | Erzeugt Actor und Critic passend zur Latent-Größe |

**Ablauf von `iterative_train`:**

1. 25 Zufallsepisoden sammeln
2. Weltmodell vortrainieren (30 Epochen)
3. Kurzes Actor-Critic-Pretraining (3 Iterationen)
4. Hauptschleife (200 Iterationen): Daten sammeln → Weltmodell trainieren → Actor/Critic trainieren → evaluieren
5. Bestes Modell (nach `eval_return`) laden und zurückgeben

---

## Hyperparameter

Zentrale Einstellungen in `main.py` (`TrainConfig`) und `train.py` (Konstanten).

**`TrainConfig` (Auszug):**

| Parameter | Wert | Bedeutung |
|---|---|---|
| `iterations` | 200 | Anzahl der Hauptiterationen |
| `batch_size` / `seq_len` | 32 / 25 | Batch- und Sequenzlänge im Weltmodell-Training |
| `replay_capacity` | 2000 | Max. Episoden im Buffer |
| `imagination_horizon` | 15 | Länge der geträumten Trajektorien |
| `entropy_start` → `entropy_end` | 6e-3 → 1e-3 | Linear annealte Actor-Entropie |
| `ac_updates_per_iter` | 10 | Actor-Critic-Updates pro Iteration |
| `collect_episodes_per_iter` | 6 | Neue Episoden pro Iteration |
| `eval_episodes` | 10 | Episoden pro Evaluation |
| `max_steps` | 500 | Maximale Episodenlänge |

**Learning Rates:** Weltmodell `2e-4`, Actor `4e-5`, Critic `1e-4`.

**`train.py`-Konstanten:** `GAMMA=0.995`, `LAMBDA=0.95`, `KL_SCALE=0.1`, `ALPHA_KL=0.8` (KL-Balancing), `SLOW_CRITIC_UPDATE=50`.

---

## Analyse-Skripte

Alle Plots werden in `cartpole/plots/` gespeichert, mit Suffix `_2d` bzw. `_4d`.

### `analysis_state_model.py`

Bewertet das Weltmodell und die Policy:

| Output | Inhalt |
|---|---|
| `wm_accuracy_<N>d.png` | Vorhersagegenauigkeit des Weltmodells pro Zustandsvariable |
| `balance_duration_<N>d.png` | Wie lange der Agent über viele Episoden balanciert |
| `imagination_drift_<N>d.png` | Abweichung der Imagination von der echten Dynamik über den Horizont |
| `trajectory_comparison_<N>d.png` | Qualitativer Vergleich echte vs. imaginierte Trajektorie |
| `policy_landscape_<N>d.png` | Aktionswahl des Actors über den Zustandsraum |

### `latent_analysis.py`

Untersucht, was der Latentraum kodiert. Ein separates Probing-Netz versucht, die 4 echten Zustandsvariablen (`ep["fulls"]`) aus `[z, h]` zu rekonstruieren. Das ist besonders interessant für das **2D-Modell**: Werden die nicht beobachteten Geschwindigkeiten (ẋ, θ̇) im Latentraum implizit erlernt?

| Output | Inhalt |
|---|---|
| `decoder_mse_scores_<N>d.png` | MSE des Probing-Decoders pro Zustandsvariable |
| `latent_pca_<N>d.png` | PCA des Latentraums, eingefärbt nach Zustandswerten |
| `reconstruction_<N>d.png` | Rekonstruktion der Beobachtungen |
| `prior_rollout_<N>d.png` | Prior-Rollout ohne Beobachtungen |

---

## Tests

| Datei | Zweck | Dauer |
|---|---|---|
| `sanity_tests.py` | Shapes, endliche Losses, Gradienten, ReplayBuffer | schnell |
| `performance_tests.py` | Loss sinkt, Actor-Update verschlechtert nicht stark, Inferenz- und Imagination-Laufzeit | langsamer |

Alle Tests laufen für **2D und 4D** (parametrisierte Fixtures).

---

## Referenzen

- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2), 2021
- [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
