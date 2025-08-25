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

from src.utils.file_manager import read_jsonl
from src.utils.taxonomy import build_taxonomy_maps
from src.data import make_examples, PretrainDataset, Collator
from src.models import BusinessBERT2Pretrain
from src.training.engine import run_eval
from src.training.metrics import mlm_accuracy, binary_accuracy, top1_accuracy


def set_seed(seed: int):
    random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, data_override: str = None) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    if data_override:
        config["jsonl_path"] = data_override
    return config


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Parse cli args
    parser = argparse.ArgumentParser(description="BusinessBERT 2.0 – Pretraining")
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument("--data", required=False, help="Optional: override jsonl path")
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "..", "..", "config", "pretrain.yaml")
        print(f"No config path specified, using default: {args.config}")

    if args.data is None:
        args.data = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample.jsonl")
        print(f"No data path specified, using default: {args.data}")

    # Load config from config/pretrain.yaml and set up
    config = load_config(args.config, args.data)
    os.makedirs(config["save_dir"], exist_ok=True)

    # Set random seed
    set_seed(int(config.get("seed", 42)))

    # ---------------- Data ----------------
    dataset = read_jsonl(config["jsonl_path"])
    print(f"Loaded {len(dataset)} rows")

    random.shuffle(dataset)

    # IC specific – build taxonomy maps
    taxonomy_maps = build_taxonomy_maps(dataset, config["field_sic2"], config["field_sic3"], config["field_sic4"])
    print(
        f"SIC sizes: "
        f"2-digit={len(taxonomy_maps['sic2_list'])}, "
        f"3-digit={len(taxonomy_maps['sic3_list'])}, "
        f"4-digit={len(taxonomy_maps['sic4_list'])}"
    )

    # Split train/val
    n_total = len(dataset)
    n_val = max(1, int(n_total * config["val_ratio"]))
    val_rows = dataset[:n_val]
    train_rows = dataset[n_val:]

    # Create examples (sentence pairs with SOP labels and SIC labels)
    train_examples = make_examples(train_rows, config["field_text"], config["field_sic2"], config["field_sic3"], config["field_sic4"])
    val_examples   = make_examples(val_rows,   config["field_text"], config["field_sic2"], config["field_sic3"], config["field_sic4"])

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else "[PAD]"

    # Create train and val datasets
    train_dataset = PretrainDataset(train_examples, tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])
    val_dataset   = PretrainDataset(val_examples,   tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])

    # Create data loaders
    collate = Collator(
        tokenizer=tokenizer,
        mlm_probability=config["mlm_probability"],
        rand_probability=config["random_token_probability"],
        keep_probability=config["keep_token_probability"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate,
        drop_last=False,
    )

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_config = BertConfig.from_pretrained(config["base_tokenizer"])
    model = BusinessBERT2Pretrain(
        config=bert_config,
        n_sic2=len(taxonomy_maps["sic2_list"]),
        n_sic3=len(taxonomy_maps["sic3_list"]),
        n_sic4=len(taxonomy_maps["sic4_list"]),
        A32=taxonomy_maps["A32"].to(device) if len(taxonomy_maps["sic3_list"]) and len(taxonomy_maps["sic2_list"]) else torch.empty(0),
        A43=taxonomy_maps["A43"].to(device) if len(taxonomy_maps["sic4_list"]) and len(taxonomy_maps["sic3_list"]) else torch.empty(0),
        loss_weights=config["loss_weights"],
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # -------- Training loop --------
    total_train_steps = config["num_train_epochs"] * max(1, len(train_loader))
    global_step = 0

    for epoch in range(1, config["num_train_epochs"] + 1):
        model.train()
        running = collections.defaultdict(float)
        counts = collections.defaultdict(int)
        t0 = time.time()

        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out["loss"]

            # TODO: Warm up consistency weight linearly over first 30% of total steps
            progress = min(1.0, global_step / max(1, int(0.3 * total_train_steps)))
            model.loss_weights["consistency"] = float(config["loss_weights"].get("consistency", 0.1)) * progress

            # TODO
            scaler.scale(loss).backward()
            if config.get("grad_clip", 0):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            for k, v in out["losses"].items():
                running[f"loss_{k}"] += float(v.item())
                counts[f"loss_{k}"] += 1

            correct, total = mlm_accuracy(out["mlm_logits"], batch["mlm_labels"])
            running["acc_mlm_correct"] += correct; running["acc_mlm_total"] += total

            correct, total = binary_accuracy(out["sop_logits"], batch["sop_labels"])
            running["acc_sop_correct"] += correct; running["acc_sop_total"] += total

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

            if step % max(1, config["logging_steps"] // 5) == 0 or step == 1:
                progress_bar.set_postfix({
                    "loss_mlm": f"{(running['loss_mlm']/max(1, counts['loss_mlm'])):.3f}" if counts.get('loss_mlm',0) else "-",
                    "loss_sop": f"{(running['loss_sop']/max(1, counts['loss_sop'])):.3f}" if counts.get('loss_sop',0) else "-",
                    "loss_ic4": f"{(running['loss_ic4']/max(1, counts['loss_ic4'])):.3f}" if counts.get('loss_ic4',0) else "-",
                    "cons": f"{(running['loss_consistency']/max(1, counts['loss_consistency'])):.3f}" if counts.get('loss_consistency',0) else "-",
                    "acc_sop": f"{(running['acc_sop_correct']/max(1, running['acc_sop_total'])):.3f}" if running.get('acc_sop_total',0) else "-",
                })

            if global_step % config["logging_steps"] == 0:
                msg = [f"epoch {epoch} step {global_step}"]
                for key in ["mlm", "sop", "ic2", "ic3", "ic4", "consistency"]:
                    loss_key = f"loss_{key}"
                    if counts.get(loss_key, 0):
                        msg.append(f"{loss_key}:{running[loss_key]/counts[loss_key]:.4f}")
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

            progress_bar.update(1)

        progress_bar.close()
        dt = time.time() - t0
        print(f"Epoch {epoch} finished in {dt/60:.1f} min")
        run_eval(model, device, val_loader, desc=f"VAL epoch {epoch}")

    # ---------------- Save ----------------
    model.save_pretrained(config["save_dir"], safe_serialization=bool(config.get("safe_serialization", False)))
    tokenizer.save_pretrained(config["save_dir"])
    print(f"Saved to {config['save_dir']}")


if __name__ == "__main__":
    main()