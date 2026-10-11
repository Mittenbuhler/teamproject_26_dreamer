# DreamerV2 Reinforcement Learning – CartPole & MinAtar Breakout

Dieses Projekt enthält eine Implementierung des DreamerV2-Algorithmus (Modellbasiertes Reinforcement Learning mit diskreten Weltmodellen) für zwei verschiedene Umgebungen:

1. CartPole (`cartpole/`): Vektorbasierte Beobachtungen (2D oder 4D), perfekt geeignet für theoretische Analysen und Latent-Space-Probing.
2. MinAtar Breakout (`minatar_breakout/`): Pixelbasierte Bildbeobachtungen (10x10x4 Grid) mit einem Convolutional Neural Network (CNN) als Encoder/Decoder im Weltmodell.

---

## Projektstruktur

```text
.
├── cartpole/             # Vektor-basiertes DreamerV2 für CartPole-v1
└── minatar_breakout/     # Bild-basiertes DreamerV2 mit CNN-RSSM für MinAtar Breakout
