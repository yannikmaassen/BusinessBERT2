from __future__ import annotations
import argparse, os, random
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from tqdm.auto import tqdm

from src.utils.io import load_yaml, read_jsonl, ensure_dir


class SentencesDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], text_field: str, tokenizer, max_len: int):
        self.examples = []
        for r in rows:
            sents = r.get(text_field, [])
            if isinstance(sents, list) and len(sents) > 0:
                # join to a short snippet for the smoke test
                txt = " ".join(sents[:3])
                self.examples.append(txt)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tok(
            self.examples[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="Optional override for jsonl path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.data:
        cfg["jsonl_path"] = args.data

    set_seed(cfg.get("seed", 42))

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])  # bert-base-uncased
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

    rows = read_jsonl(cfg["jsonl_path"])[:200]  # tiny slice for quick smoke
    ds = SentencesDataset(rows, cfg["text_field"], tok, cfg["max_seq_len"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    # Minimal model: BERT encoder + a tiny MLM-like projection for smoke (not full MLM)
    enc = BertModel.from_pretrained(cfg["model_name"]).to(device)
    head = nn.Linear(enc.config.hidden_size, enc.config.hidden_size).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=cfg["learning_rate"])

    total_steps = min(cfg.get("num_steps", 20), len(dl))
    pbar = tqdm(total=total_steps, desc="smoke-pretrain")

    enc.train(); head.train()
    for step, batch in enumerate(dl, start=1):
        if step > total_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = enc(**batch, return_dict=True)
        # take CLS as pooled vector and pass through a small head
        cls = out.last_hidden_state[:, 0, :]  # [B,H]
        proj = head(cls)                       # [B,H]
        # dummy target: L2 toward zero vector (just to exercise optimizer)
        loss = (proj ** 2).mean()
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "bs": batch["input_ids"].size(0)})
        pbar.update(1)

    pbar.close()

    # Save encoder (tokenizer optional here)
    ensure_dir(cfg["save_dir"])
    enc.save_pretrained(cfg["save_dir"])
    tok.save_pretrained(cfg["save_dir"])
    print(f"Saved encoder+tokenizer to {cfg['save_dir']}")


if __name__ == "__main__":
    main()