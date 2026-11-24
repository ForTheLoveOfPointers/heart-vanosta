"""Tests for the TinyCNN embedding used by hybrid models."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase0.models.cnn import TinyCNN, TinyCNNEmbedder, embed_cnn


def _model():
    torch.manual_seed(0)
    return TinyCNN(in_channels=1, context_dim=3, width=4, input_length=64)


def test_embedder_shape_and_finiteness():
    model = _model()
    embedder = TinyCNNEmbedder(model)
    x = torch.randn(5, 1, 64)
    ctx = torch.randn(5, 3)
    z = embedder(x, ctx)
    assert z.shape == (5, 4 * 4 * 2 + 3)
    assert torch.isfinite(z).all()


def test_embed_cnn_matches_embedder_on_subset():
    model = _model()
    signals = np.random.default_rng(0).normal(size=(12, 1, 64)).astype(np.float32)
    context = np.random.default_rng(1).normal(size=(12, 3)).astype(np.float32)
    idx = np.array([2, 5, 9])
    z = embed_cnn(model, signals, context, indices=idx, batch_size=4)
    embedder = TinyCNNEmbedder(model).eval()
    with torch.no_grad():
        ref = embedder(
            torch.from_numpy(signals[idx]), torch.from_numpy(context[idx])
        ).numpy()
    assert z.shape == (3, 35)
    assert np.allclose(z, ref, atol=1e-5)
