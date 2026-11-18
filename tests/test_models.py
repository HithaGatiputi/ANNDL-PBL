"""Model tests. All tensors here are RANDOM / SYNTHETIC (testing only)."""
import copy

import torch

from src.models.rc_dgnn import RCDGNN
from src.utils.config import load_config


def _batch(B=8, L=30, N=5, F=3, s=28, e=19, g=8, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    return {"X": torch.randn(B, L, N, F, generator=g_), "S": torch.randn(B, L, s, generator=g_),
            "E": torch.randn(B, L, e, generator=g_), "G": torch.randn(B, g, generator=g_),
            "y": (torch.rand(B, generator=g_) > 0.5).float()}


def test_forward_shapes_and_row_sums():
    for variant in ["rcdgnn_daily", "static_gnn"]:
        cfg = load_config(variant)
        m = RCDGNN(cfg, 3, 28, 19, 8)
        out = m(_batch())
        assert out["logit"].shape == (8,)
        assert out["A_last"].shape == (8, 5, 5)
        torch.testing.assert_close(out["A_last"].sum(-1), torch.ones(8, 5))
        n = sum(p.numel() for p in m.parameters())
        print(f"{variant}: {n} parameters")
        assert n < 100_000


def test_static_variant_has_no_event_graph_change():
    cfg = load_config("static_gnn")
    m = RCDGNN(cfg, 3, 28, 19, 8).eval()
    b1, b2 = _batch(seed=1), _batch(seed=1)
    b2["S"] = b2["S"] * 10  # signature must not affect the static graph
    torch.testing.assert_close(m(b1)["A_last"], m(b2)["A_last"])


def test_overfit_64_samples():
    """The model can memorise 64 random samples (>95% train accuracy): proves it can learn."""
    torch.manual_seed(0)
    cfg = copy.deepcopy(load_config("rcdgnn_daily"))
    cfg["model"]["dropout"] = 0.0
    cfg["model"]["gnn_dropout"] = 0.0
    m = RCDGNN(cfg, 3, 28, 19, 8)
    b = _batch(B=64, seed=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(400):
        out = m(b)
        loss = lossf(out["logit"], b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((m(b)["logit"] > 0).float() == b["y"]).float().mean().item()
    print(f"overfit train acc {acc:.3f}")
    assert acc > 0.95
