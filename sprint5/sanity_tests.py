import torch
import numpy as np

from sprint5.data import ReplayBuffer, SequenceDataset
from sprint5.models import RSSM, Actor
from sprint5.train import compute_world_model_loss


def test_replay_buffer():
    rb = ReplayBuffer(capacity=2)
    rb.add_episode({"actions": [1, 2]})
    rb.add_episode({"actions": [1, 2]})
    assert len(rb) == 2
    rb.add_episode({"actions": [1, 2]})
    assert len(rb) == 2


def test_sequence_dataset_shapes():
    ep = {
        "vis": [[0, 0], [1, 1], [2, 2]],
        "actions": [[1, 0], [0, 1]],
        "rewards": [[1], [1]],
        "continues": [[1], [0]],
    }
    dataset = SequenceDataset([ep], seq_len=5)
    sample = dataset[0]
    assert sample["observations"].shape == (6, 2)
    assert sample["actions"].shape == (5, 2)
    assert sample["rewards"].shape == (5, 1)
    assert sample["continues"].shape == (5, 1)


def test_rssm_forward_shapes():
    model = RSSM()
    observations = torch.randn(4, 17, 4)
    actions = torch.randn(4, 16, 2)
    out = model.observe_forward(observations, actions)
    assert out["prior_logits"].shape == (4, 16, model.C, model.K)
    assert out["posterior_logits"].shape == (4, 16, model.C, model.K)
    assert out["prior_predictions"]["observation"].shape == (4, 16, 4)
    assert out["posterior_predictions"]["reward"].shape == (4, 16, 1)


def test_imagination_shapes():
    model = RSSM()
    start_z, start_h = model.initial(batch_size=3)
    actor = Actor(model.stoch_size + model.h_size)
    imag = model.imagine(start_z, start_h, actor, horizon=10)
    assert imag["features"].shape == (3, 10, model.stoch_size + model.h_size)
    assert imag["rewards"].shape == (3, 10, 1)
    assert imag["discounts"].shape == (3, 10, 1)


def test_world_model_loss_is_finite():
    model = RSSM()
    observations = torch.randn(2, 17, 4)
    actions = torch.randn(2, 16, 2)
    rewards = torch.zeros(2, 16, 1)
    continues = torch.ones(2, 16, 1)
    out = model.observe_forward(observations, actions)
    loss, metrics = compute_world_model_loss(out, observations, rewards, continues)
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["total_loss"])
    assert np.isfinite(metrics["kl"])


def test_gradients_exist():
    model = RSSM()
    observations = torch.randn(2, 17, 4)
    actions = torch.randn(2, 16, 2)
    rewards = torch.zeros(2, 16, 1)
    continues = torch.ones(2, 16, 1)
    out = model.observe_forward(observations, actions)
    loss, _ = compute_world_model_loss(out, observations, rewards, continues)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(gradients) > 0
    total_norm = sum(g.abs().sum().item() for g in gradients)
    assert total_norm > 0