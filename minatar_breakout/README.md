# MinAtar Breakout – DreamerV2 mit CNN-RSSM-Weltmodell

Dieser Ordner (`minatar_breakout/`) enthält unsere Implementierung eines DreamerV2-Agenten für das MinAtar-Spiel Breakout. Im Gegensatz zur CartPole-Variante arbeitet dieses Modell direkt auf Bild-Beobachtungen (10x10x4 Pixel-Grids) anstelle von flachen Vektoren.

Das Weltmodell verwendet einen CNN-Encoder, um die Bilder in den latenten Raum zu komprimieren, und einen CNN-Decoder (mit Binary Cross-Entropy Loss), um die Bilddaten während des Trainings zu rekonstruieren. Actor und Critic lernen wie gewohnt im geträumten latenten Raum.

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Ordnerstruktur](#2-ordnerstruktur)
3. [Schnellstart](#3-schnellstart)
4. [Funktionsweise & Besonderheiten gegenüber CartPole](#4-funktionsweise--besonderheiten-gegenüber-cartpole)
5. [Dateien im Detail](#5-dateien-im-detail)
6. [Analysen & Tests](#6-analysen--tests)

---

## 1. Überblick

| Aspekt | Beschreibung |
|---|---|
| Umgebung | MinAtar Breakout (native API, max. 1000 Schritte) |
| Algorithmus | DreamerV2 (diskrete Latents, CNN-RSSM, REINFORCE-Actor, λ-Returns) |
| Beobachtungen | 10x10x4 bool/float Grid (Kanäle: `paddle`, `ball`, `trail`, `brick`) |
| Aktionen | 3 diskrete Aktionen (`left`, `right`, `no-op`), One-Hot kodiert |
| Sprache im Code | Kommentare und Docstrings auf Deutsch |

**Grundidee:**

1. Zufällige Episoden sammeln (mit nativen MinAtar-Grids) → Replay Buffer.
2. CNN-Weltmodell (RSSM) auf Bildsequenzen aus dem Buffer trainieren (MSE + BCE für Bilder + KL-Divergenz).
3. Actor/Critic in der Imagination des Weltmodells trainieren (nur Latent-Raum, kein Dekodieren notwendig).
4. Mit dem Actor neue Episoden in der echten Umgebung sammeln → zurück zu Schritt 2.

---

## 2. Ordnerstruktur

```text
minatar_breakout/
├── __init__.py                  # Modul-Init
├── models_breakout.py           # RSSM mit MinAtarEncoder & MinAtarDecoder, Actor, Critic
├── data_breakout.py             # ReplayBuffer, SequenceDataset, nativer MinAtar-Collector
├── train_breakout.py            # Weltmodell-Loss (mit Bild-BCE), λ-Return, Actor-Critic
├── main_breakout.py             # Trainingsschleife, Evaluation, Speichern der Gewichte
├── analysis_breakout.py         # Analyse-Skript (WM-Genauigkeit, Rekonstruktion, Prior-Rollout, Reward)
├── learning_tests_breakout.py   # Erweiterte Unit- & Lerntests (Loss sinkt, Pixel-Acc steigt)
├── world_model_breakout.pth     # (wird erzeugt) trainierte Gewichte
├── actor_breakout.pth           # (wird erzeugt) trainierte Gewichte
├── critic_breakout.pth          # (wird erzeugt) trainierte Gewichte
├── training_log_breakout.npy    # (wird erzeugt) Verlauf der Evaluierungs-Returns
└── plots/                       # (wird erzeugt) Analyse-Plots als PNGs
```

---

## 3. Schnellstart

> **Wichtig:** Alle Befehle werden aus dem **Projekt-Root** (Ordner über `minatar_breakout/`) ausgeführt, da relative Importe verwendet werden (`python -m minatar_breakout.<modul>`).

### 3.1 Trainieren

```bash
python3 -m minatar_breakout.main_breakout
```

Speichert die Gewichte direkt im Ordner (`world_model_breakout.pth`, `actor_breakout.pth`, `critic_breakout.pth`) und das Trainingslog (`training_log_breakout.npy`).

### 3.2 Analysen ausführen

Nach dem erfolgreichen Training können Visualisierungen (wie Bild-Rekonstruktionen und Prior-Rollouts) erzeugt werden:

```bash
python -m minatar_breakout.analysis_breakout
```

Die Ergebnisse landen in `minatar_breakout/plots/`.

### 3.3 Tests ausführen

Um zu überprüfen, ob das Modell messbar lernt (z. B. sinkender Rekonstruktions-Loss, steigende Pixel-Genauigkeit):

```bash
python3 -m pytest minatar_breakout/learning_tests_breakout.py -v -s
```

---

## 4. Funktionsweise & Besonderheiten gegenüber CartPole

- **Bildverarbeitung:** Statt eines 4D-Zustandsvektors verarbeitet der `MinAtarEncoder` einen `(4, 10, 10)`-Tensor. Der Posterior nutzt das erzeugte Embedding anstelle des Rohzustands.
- **Loss-Funktion:** Da die Bildkanäle binär bzw. Float-Wahrscheinlichkeiten sind, wird für die Beobachtungsrekonstruktion Binary Cross-Entropy (`F.binary_cross_entropy_with_logits`) anstelle von einfachem MSE verwendet.
- **Discount-Faktor (γ):** Bei Atari-Spielen wie Breakout plant der Agent weiter in die Zukunft (γ = 0.99 statt 0.995 bei CartPole).
- **Aktionsraum:** Breakout nutzt 3 diskrete Aktionen (links, rechts, nichts tun). Die Entropie der Policy initialisiert sich entsprechend auf ln(3) ≈ 1.10.

---

## 5. Dateien im Detail

| Datei | Beschreibung |
|---|---|
| `models_breakout.py` | Enthält den `MinAtarEncoder` (CNN für 10x10x4-Grids) und den `MinAtarDecoder` (Transposed Convolutions für die Bildrekonstruktion) |
| `train_breakout.py` | Beinhaltet die angepasste Loss-Berechnung (`compute_world_model_loss`), die den Bildfehler pixelweise über die Kanäle mittelt und per BCE gewichtet |
| `analysis_breakout.py` | Generiert 5 Analyse-Plots (Weltmodell-Genauigkeit, kanalgenaue Bildrekonstruktion, Prior-Rollout, Reward-Verteilung und Lernkurve) |

---

## 6. Analysen & Tests

| Datei | Zweck |
|---|---|
| `analysis_breakout.py` | Erzeugt die 5 Analyse-Plots in `minatar_breakout/plots/` (siehe [3.2](#32-analysen-ausführen)) |
| `learning_tests_breakout.py` | Prüft, ob der Rekonstruktions-Loss sinkt und die Pixel-Genauigkeit steigt (siehe [3.3](#33-tests-ausführen)) |
