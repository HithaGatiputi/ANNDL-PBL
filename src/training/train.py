"""Train one variant (all seeds), save predictions, metrics and loss curves.

Usage: python -m src.training.train --config baseline_lstm

Neural models: BCEWithLogitsLoss, AdamW, early stopping on validation log loss (patience from
config), best checkpoint kept, MC-dropout predictions (T samples) -> p_up (mean) and p_std (std).
Outputs: reports/results/<variant>/seed<k>/preds_{val,test}.csv and metrics.json.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import metrics_by_subset
from src.models.baselines import LogRegBaseline, LSTMBaseline, MajorityBaseline
from src.training.dataset import DayData, build_dataset
from src.utils import paths
from src.utils.config import load_config
from src.utils.seed import set_seed


def to_batch(d: DayData, idx: np.ndarray) -> dict[str, torch.Tensor]:
    """Slice the arrays at `idx` into a dict of tensors."""
    return {"X": torch.from_numpy(d.X[idx]), "S": torch.from_numpy(d.S[idx]),
            "E": torch.from_numpy(d.E[idx]), "G": torch.from_numpy(d.G[idx]),
            "y": torch.from_numpy(d.y[idx])}


def build_model(cfg: dict, d: DayData) -> nn.Module:
    """Create the neural model named in cfg['model']['type']."""
    m = cfg["model"]
    if m["type"] == "lstm":
        e_dim = d.E.shape[2] if m["use_event_emb"] else 0
        return LSTMBaseline(d.X.shape[3], m["lstm_hidden"], m["dropout"], e_dim)
    if m["type"] == "rcdgnn":
        from src.models.rc_dgnn import RCDGNN
        return RCDGNN(cfg, f_dim=d.X.shape[3], s_dim=d.S.shape[2], e_dim=d.E.shape[2], g_dim=d.G.shape[1])
    raise ValueError(m["type"])


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def eval_logloss(model: nn.Module, batch: dict) -> float:
    """Deterministic (dropout off) validation log loss."""
    model.eval()
    out = model(batch)
    return float(nn.functional.binary_cross_entropy_with_logits(out["logit"], batch["y"]))


@torch.no_grad()
def mc_predict(model: nn.Module, batch: dict, T: int) -> tuple[np.ndarray, np.ndarray]:
    """MC dropout: dropout layers active, T forward passes -> mean and std of P(up)."""
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    probs = torch.stack([torch.sigmoid(model(batch)["logit"]) for _ in range(T)])
    model.eval()
    return probs.mean(0).numpy(), probs.std(0).numpy()


def train_neural(cfg: dict, d: DayData, tr: np.ndarray, va: np.ndarray, seed: int,
                 y_train_override: np.ndarray | None = None, verbose: bool = False):
    """Train one seed. Returns (best model, history dict)."""
    set_seed(seed)
    t = cfg["train"]
    model = build_model(cfg, d)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    lossf = nn.BCEWithLogitsLoss()
    train_b, val_b = to_batch(d, tr), to_batch(d, va)
    if y_train_override is not None:
        train_b["y"] = torch.from_numpy(y_train_override.astype(np.float32))
    n = len(tr)
    best, best_state, wait = np.inf, None, 0
    hist = {"train": [], "val": []}
    g = torch.Generator().manual_seed(seed)
    for epoch in range(t["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)  # shuffling WITHIN the train split only
        tot = 0.0
        for i in range(0, n, t["batch_size"]):
            bi = perm[i:i + t["batch_size"]]
            b = {k: v[bi] for k, v in train_b.items()}
            out = model(b)
            loss = lossf(out["logit"], b["y"])
            opt.zero_grad()
            (loss + out["penalty"]).backward()
            opt.step()
            tot += float(loss.detach()) * len(bi)
        vl = eval_logloss(model, val_b)
        hist["train"].append(tot / n)
        hist["val"].append(vl)
        if verbose:
            print(f"  epoch {epoch:2d} train {tot / n:.4f} val {vl:.4f}")
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= t["early_stop"]:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val"]))
    return model, hist


def save_preds(out_dir, meta: pd.DataFrame, idx: np.ndarray, p: np.ndarray, s: np.ndarray, name: str) -> pd.DataFrame:
    """Write preds_<name>.csv with date, y, p_up, p_std (+ subset flags for evaluation)."""
    df = meta.iloc[idx][["date", "y", "is_event_day", "is_post_event_day", "regime"]].copy()
    df["p_up"], df["p_std"] = p, s
    df.to_csv(out_dir / f"preds_{name}.csv", index=False)
    return df


def plot_losses(histories: list[dict], variant: str) -> None:
    """Train/val loss curves for every seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, h in enumerate(histories):
        ax.plot(h["train"], color="C0", alpha=0.6, label="train" if k == 0 else None)
        ax.plot(h["val"], color="C1", alpha=0.6, label="val" if k == 0 else None)
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title(f"{variant}: loss per seed"); ax.legend()
    plt.tight_layout(); plt.savefig(paths.FIGURES / f"{variant}_loss.png", dpi=110); plt.close(fig)


def run_variant(cfg: dict) -> dict:
    """Train/evaluate all seeds of one variant and save everything."""
    variant = cfg["variant"]
    d = build_dataset(cfg)
    split = d.meta["split"].values
    tr, va, te = (np.where(split == s)[0] for s in ["train", "val", "test"])
    root = paths.RESULTS / variant
    mtype = cfg["model"]["type"]
    seeds = cfg["train"]["seeds"] if mtype not in ("majority", "logreg") else [0]  # deterministic models
    histories, summary = [], {}
    for seed in seeds:
        out_dir = root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        info: dict = {"seed": seed}
        if mtype == "majority":
            m = MajorityBaseline().fit(d.y[tr])
            pv, pt = m.predict_proba(len(va)), m.predict_proba(len(te))
            sv, st = np.zeros(len(va)), np.zeros(len(te))
        elif mtype == "logreg":
            m = LogRegBaseline(cfg["train"]["logreg_C"], cfg["train"]["logreg_lags"]).fit(d.X[tr], d.G[tr], d.y[tr])
            pv, pt = m.predict_proba(d.X[va], d.G[va]), m.predict_proba(d.X[te], d.G[te])
            sv, st = np.zeros(len(va)), np.zeros(len(te))
        else:
            model, hist = train_neural(cfg, d, tr, va, seed)
            histories.append(hist)
            T = cfg["model"]["mc_samples"]
            pv, sv = mc_predict(model, to_batch(d, va), T)
            pt, st = mc_predict(model, to_batch(d, te), T)
            info.update(n_params=n_params(model), best_epoch=hist["best_epoch"], epochs_run=len(hist["val"]))
            torch.save(model.state_dict(), out_dir / "model.pt")
        dv = save_preds(out_dir, d.meta, va, pv, sv, "val")
        dt = save_preds(out_dir, d.meta, te, pt, st, "test")
        n_bins = cfg["eval"]["ece_bins"]
        info["val"] = metrics_by_subset(dv, n_bins)
        info["test"] = metrics_by_subset(dt, n_bins)
        (out_dir / "metrics.json").write_text(json.dumps(info, indent=2, default=float))
        summary[seed] = info
        print(f"[{variant}] seed {seed}: val acc {info['val']['all']['acc']:.3f} logloss {info['val']['all']['logloss']:.4f} | "
              f"test acc {info['test']['all']['acc']:.3f} mcc {info['test']['all']['mcc']:.3f}"
              + (f" | params {info['n_params']} best_ep {info['best_epoch']}" if "n_params" in info else ""), flush=True)
    if histories:
        plot_losses(histories, variant)
    accs = [s["test"]["all"]["acc"] for s in summary.values()]
    print(f"[{variant}] test acc mean {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="variant name, e.g. baseline_lstm")
    args = ap.parse_args()
    torch.set_num_threads(4)
    run_variant(load_config(args.config))


if __name__ == "__main__":
    main()
