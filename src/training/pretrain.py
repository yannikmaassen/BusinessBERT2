import argparse
import collections
import os
import random
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from transformers import AutoTokenizer, BertConfig

from ..utils.file_manager import read_jsonl
from ..utils.taxonomy import build_taxonomy_maps
from ..data import make_examples, PretrainDataset, Collator
from ..models import BusinessBERT2Pretrain
from .engine import run_eval


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, data_override: str = None) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if data_override:
        cfg["jsonl_path"] = data_override
    return cfg


def main():
    parser = argparse.ArgumentParser(description="BusinessBERT 2.0 – Pretraining")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", required=False, help="Optional: override jsonl path")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))

    # ---------------- Data ----------------
    rows_all = read_jsonl(cfg["jsonl_path"])
    print(f"Loaded {len(rows_all)} rows")

    random.shuffle(rows_all)

    maps = build_taxonomy_maps(rows_all, cfg["field_sic2"], cfg["field_sic3"], cfg["field_sic4"])
    print(
        f"SIC sizes: 2-digit={len(maps['sic2_list'])}, "
        f"3-digit={len(maps['sic3_list'])}, 4-digit={len(maps['sic4_list'])}"
    )

    n_total = len(rows_all)
    n_val = max(1, int(n_total * cfg["val_ratio"]))
    val_rows = rows_all[:n_val]
    train_rows = rows_all[n_val:]

    train_exs = make_examples(train_rows, cfg["field_text"], cfg["field_sic2"], cfg["field_sic3"], cfg["field_sic4"])
    val_exs   = make_examples(val_rows,   cfg["field_text"], cfg["field_sic2"], cfg["field_sic3"], cfg["field_sic4"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else "[PAD]"

    train_ds = PretrainDataset(train_exs, tokenizer, cfg["max_seq_len"], maps["idx2"], maps["idx3"], maps["idx4"])
    val_ds   = PretrainDataset(val_exs,   tokenizer, cfg["max_seq_len"], maps["idx2"], maps["idx3"], maps["idx4"])

    collate = Collator(
        tokenizer=tokenizer,
        mlm_prob=cfg["mlm_probability"],
        rand_prob=cfg["random_token_probability"],
        keep_prob=cfg["keep_token_probability"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        drop_last=False,
    )

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BertConfig.from_pretrained(cfg["base_tokenizer"])
    model = BusinessBERT2Pretrain(
        config=config,
        n_sic2=len(maps["sic2_list"]),
        n_sic3=len(maps["sic3_list"]),
        n_sic4=len(maps["sic4_list"]),
        A32=maps["A32"].to(device) if len(maps["sic3_list"]) and len(maps["sic2_list"]) else torch.empty(0),
        A43=maps["A43"].to(device) if len(maps["sic4_list"]) and len(maps["sic3_list"]) else torch.empty(0),
        loss_weights=cfg["loss_weights"],
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # -------- Training loop --------
    total_train_steps = cfg["epochs"] * max(1, len(train_loader))
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = collections.defaultdict(float)
        counts = collections.defaultdict(int)
        t0 = time.time()

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out["loss"]

            # Warm up consistency weight linearly over first 30% of total steps
            progress = min(1.0, global_step / max(1, int(0.3 * total_train_steps)))
            model.loss_weights["consistency"] = float(cfg["loss_weights"].get("consistency", 0.1)) * progress

            scaler.scale(loss).backward()
            if cfg.get("grad_clip", 0):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            for k, v in out["losses"].items():
                running[f"loss_{k}"] += float(v.item())
                counts[f"loss_{k}"] += 1

            from .metrics import mlm_accuracy, binary_accuracy, top1_accuracy
            c, t = mlm_accuracy(out["mlm_logits"], batch["mlm_labels"])
            running["acc_mlm_correct"] += c; running["acc_mlm_total"] += t

            c, t = binary_accuracy(out["sop_logits"], batch["sop_labels"])
            running["acc_sop_correct"] += c; running["acc_sop_total"] += t

            if out["ic2_logits"] is not None:
                c, t = top1_accuracy(out["ic2_logits"], batch["sic2"])
                running["acc_ic2_correct"] += c; running["acc_ic2_total"] += t
            if out["ic3_logits"] is not None:
                c, t = top1_accuracy(out["ic3_logits"], batch["sic3"])
                running["acc_ic3_correct"] += c; running["acc_ic3_total"] += t
            if out["ic4_logits"] is not None:
                c, t = top1_accuracy(out["ic4_logits"], batch["sic4"])
                running["acc_ic4_correct"] += c; running["acc_ic4_total"] += t

            global_step += 1

            if step % max(1, cfg["log_every"] // 5) == 0 or step == 1:
                pbar.set_postfix({
                    "loss_mlm": f"{(running['loss_mlm']/max(1, counts['loss_mlm'])):.3f}" if counts.get('loss_mlm',0) else "-",
                    "loss_sop": f"{(running['loss_sop']/max(1, counts['loss_sop'])):.3f}" if counts.get('loss_sop',0) else "-",
                    "loss_ic4": f"{(running['loss_ic4']/max(1, counts['loss_ic4'])):.3f}" if counts.get('loss_ic4',0) else "-",
                    "cons": f"{(running['loss_consistency']/max(1, counts['loss_consistency'])):.3f}" if counts.get('loss_consistency',0) else "-",
                    "acc_sop": f"{(running['acc_sop_correct']/max(1, running['acc_sop_total'])):.3f}" if running.get('acc_sop_total',0) else "-",
                })

            if global_step % cfg["log_every"] == 0:
                msg = [f"epoch {epoch} step {global_step}"]
                for key in ["mlm", "sop", "ic2", "ic3", "ic4", "consistency"]:
                    lk = f"loss_{key}"
                    if counts.get(lk, 0):
                        msg.append(f"{lk}:{running[lk]/counts[lk]:.4f}")
                if running.get("acc_mlm_total", 0) > 0:
                    msg.append(f"acc_mlm:{running['acc_mlm_correct']/max(1, running['acc_mlm_total']):.4f}")
                if running.get("acc_sop_total", 0) > 0:
                    msg.append(f"acc_sop:{running['acc_sop_correct']/max(1, running['acc_sop_total']):.4f}")
                if running.get("acc_ic2_total", 0) > 0:
                    msg.append(f"acc_ic2:{running['acc_ic2_correct']/running['acc_ic2_total']:.4f}")
                if running.get("acc_ic3_total", 0) > 0:
                    msg.append(f"acc_ic3:{running['acc_ic3_correct']/running['acc_ic3_total']:.4f}")
                if running.get("acc_ic4_total", 0) > 0:
                    msg.append(f"acc_ic4:{running['acc_ic4_correct']/running['acc_ic4_total']:.4f}")
                print(" | ".join(msg))

            pbar.update(1)

        pbar.close()
        dt = time.time() - t0
        print(f"Epoch {epoch} finished in {dt/60:.1f} min")
        run_eval(model, device, val_loader, desc=f"VAL epoch {epoch}")

    # ---------------- Save ----------------
    model.save_pretrained(cfg["save_dir"], safe_serialization=bool(cfg.get("safe_serialization", False)))
    tokenizer.save_pretrained(cfg["save_dir"])
    print(f"Saved to {cfg['save_dir']}")


if __name__ == "__main__":
    main()