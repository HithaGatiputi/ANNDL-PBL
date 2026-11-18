"""Frozen FinBERT event encoder (run once, results cached).

For each event text: tokenise, split into chunks of <= 512 tokens, take the [CLS] vector of the
last hidden layer for each chunk and mean-pool across chunks (768-d). FinBERT's sentiment
probabilities (positive / negative / neutral) are averaged across chunks too.
768 -> 16 dims with PCA fitted on TRAIN events only; components are divided by their train std.
Outputs:
  data/processed/event_emb.parquet        one row per event (emb_0..15, sent_*)
  data/processed/event_emb_daily.parquet  decayed onto trading days like the signature
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features.reaction_signature import decay_to_days
from src.training.splits import split_of
from src.utils import paths
from src.utils.config import load_config


@torch.no_grad()
def embed_text(text: str, tok, model, max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean [CLS] embedding and mean sentiment probabilities over <=512-token chunks."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = max_tokens - 2  # room for [CLS] and [SEP]
    chunks = [ids[i:i + body] for i in range(0, max(len(ids), 1), body)]
    cls_vecs, probs = [], []
    for ch in chunks:
        inp = torch.tensor([[tok.cls_token_id] + ch + [tok.sep_token_id]])
        out = model(input_ids=inp, attention_mask=torch.ones_like(inp), output_hidden_states=True)
        cls_vecs.append(out.hidden_states[-1][0, 0].numpy())
        probs.append(torch.softmax(out.logits[0], -1).numpy())
    return np.mean(cls_vecs, 0), np.mean(probs, 0)


def build(cfg: dict) -> None:
    """Embed all events (cached), PCA on train, decay to days."""
    events = pd.read_csv(paths.EVENTS_CSV, parse_dates=["datetime_utc"])
    raw_file = paths.PROCESSED / "event_emb_768.parquet"
    if raw_file.exists():  # FinBERT runs once
        raw = pd.read_parquet(raw_file)
    else:
        name = cfg["text"]["model_name"]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        rows = []
        for i, ev in events.iterrows():
            v, p = embed_text(str(ev["text"]), tok, model, cfg["text"]["max_tokens"])
            rows.append({"event_id": ev["event_id"], **{f"d{j}": x for j, x in enumerate(v)},
                         **{f"sent_{l}": x for l, x in zip(labels, p)}})
            if i % 100 == 0:
                print(f"embedded {i + 1}/{len(events)}", flush=True)
        raw = pd.DataFrame(rows)
        raw.to_parquet(raw_file)
    df = raw.merge(events[["event_id", "type", "trading_day", "datetime_utc"]], on="event_id")
    dcols = [c for c in df.columns if c.startswith("d") and c[1:].isdigit()]
    train = split_of(df["trading_day"], cfg) == "train"
    k = cfg["text"]["pca_dim"]
    pca = PCA(n_components=k, random_state=0).fit(df.loc[train, dcols].values)
    Z = pca.transform(df[dcols].values)
    Z = Z / Z[train].std(0)
    emb = pd.DataFrame(Z, columns=[f"emb_{j}" for j in range(k)])
    sent = df[["sent_positive", "sent_negative", "sent_neutral"]].reset_index(drop=True)
    out = pd.concat([df[["event_id", "type", "trading_day", "datetime_utc"]].reset_index(drop=True), emb, sent], axis=1)
    assert not out.isna().any().any()
    out.to_parquet(paths.EVENT_EMB)
    meta = {"pca_fit_n_events": int(train.sum()), "pca_fit_last_day": str(df.loc[train, "trading_day"].max()),
            "explained_variance": float(pca.explained_variance_ratio_.sum())}
    (paths.PROCESSED / "event_emb_meta.json").write_text(json.dumps(meta, indent=2))
    days = pd.read_parquet(paths.INTERIM / "daily_ext.parquet").index
    daily = decay_to_days(out.drop(columns="type"), days, cfg)
    daily = daily.drop(columns=["days_since_event"])  # already in the signature
    daily.to_parquet(paths.EVENT_EMB_DAILY)
    print(f"event_emb: {out.shape}, PCA({k}) explains {meta['explained_variance']:.1%} (fit on "
          f"{meta['pca_fit_n_events']} train events, last {meta['pca_fit_last_day']}); daily cols {daily.shape[1]}")
    print(out.groupby("type")[["sent_positive", "sent_negative", "sent_neutral"]].mean().round(3).to_string())


def main() -> None:
    """CLI entry point."""
    build(load_config("base"))


if __name__ == "__main__":
    main()
