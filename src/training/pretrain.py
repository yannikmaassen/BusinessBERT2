import collections
import math
import os
import random
import time
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from transformers import AutoTokenizer, BertConfig, get_scheduler

from src.utils.file_manager import read_jsonl
from src.data import make_examples, PretrainDataset, Collator
from src.models import BusinessBERT2Pretrain
from src.training.engine import run_eval
from src.training.metrics import mlm_accuracy, binary_accuracy, top1_accuracy
from src.utils.arg_parser import parse_cli_args
from src.utils.taxonomy import build_taxonomy_maps


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, data_override: str = None) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    if data_override:
        config["jsonl_path"] = data_override
    return config


def coarse_to_fine_weights(training_step_index: int, total_training_steps: int) -> Tuple[float, float, float]:
    """
    Smooth schedule:
      Early: emphasize SIC2
      Middle: increase SIC3
      Late: emphasize SIC4
    Returns weights (weight_sic2, weight_sic3, weight_sic4) that sum to 1.0.
    """
    training_progress = min(max(training_step_index / max(1, total_training_steps), 0.0), 1.0)
    cosine_ramp = 0.5 - 0.5 * math.cos(math.pi * training_progress)  # 0 -> 1

    weight_sic2 = 0.8 * (1 - cosine_ramp) + 0.2   # 1.0 -> 0.2
    weight_sic3 = 0.3 + 0.4 * cosine_ramp         # 0.3 -> 0.7
    weight_sic4 = 0.1 + 0.9 * cosine_ramp         # 0.1 -> 1.0

    weight_sum = weight_sic2 + weight_sic3 + weight_sic4

    return weight_sic2 / weight_sum, weight_sic3 / weight_sum, weight_sic4 / weight_sum


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_cli_args()

    config = load_config(args.config, args.data)
    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    # ---------------- Data ----------------
    dataset = read_jsonl(config["jsonl_path"])
    print(f"Loaded {len(dataset)} rows")

    taxonomy_maps = build_taxonomy_maps(dataset, config["field_sic2"], config["field_sic3"], config["field_sic4"])
    print(
        f"SIC sizes: "
        f"2-digit={len(taxonomy_maps['sic2_list'])}, "
        f"3-digit={len(taxonomy_maps['sic3_list'])}, "
        f"4-digit={len(taxonomy_maps['sic4_list'])}"
    )

    train_rows, val_rows = train_test_split(
        dataset,
        test_size=config["val_ratio"],
        shuffle=True,
        random_state=config.get("seed", 42),
    )

    # Create examples (sentence pairs with SOP labels and SIC labels)
    train_examples = make_examples(train_rows, config["field_text"], config["field_sic2"], config["field_sic3"], config["field_sic4"])
    val_examples   = make_examples(val_rows,   config["field_text"], config["field_sic2"], config["field_sic3"], config["field_sic4"])

    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else "[PAD]"

    train_dataset = PretrainDataset(train_examples, tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])
    val_dataset   = PretrainDataset(val_examples,   tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])

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
    precision = str(config.get("precision", "fp32")).lower()
    device_has_cuda = torch.cuda.is_available()

    model = BusinessBERT2Pretrain(
        config=bert_config,
        n_sic2_classes=len(taxonomy_maps["sic2_list"]),
        n_sic3_classes=len(taxonomy_maps["sic3_list"]),
        n_sic4_classes=len(taxonomy_maps["sic4_list"]),
        A32=taxonomy_maps["A32"].to(device) if len(taxonomy_maps["sic3_list"]) and len(taxonomy_maps["sic2_list"]) else torch.empty(0),
        A43=taxonomy_maps["A43"].to(device) if len(taxonomy_maps["sic4_list"]) and len(taxonomy_maps["sic3_list"]) else torch.empty(0),
        loss_weights=config["loss_weights"],
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    use_amp = device_has_cuda and precision in {"fp16", "bf16"}
    amp_kwargs = {}
    if precision == "fp16":
        amp_kwargs["dtype"] = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    elif precision == "bf16":
        amp_kwargs["dtype"] = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:  # fp32
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    if device_has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[INIT] precision={precision}, use_amp={use_amp}, amp_dtype={amp_kwargs.get('dtype', 'fp32')}")

    total_train_steps = config["num_train_epochs"] * max(1, len(train_loader))

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=total_train_steps,
    )

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
            w2, w3, w4 = coarse_to_fine_weights(global_step, total_train_steps)
            base_w2 = float(config["loss_weights"].get("ic2", 1.0))
            base_w3 = float(config["loss_weights"].get("ic3", 1.0))
            base_w4 = float(config["loss_weights"].get("ic4", 1.0))
            model.loss_weights["ic2"] = base_w2 * w2
            model.loss_weights["ic3"] = base_w3 * w3
            model.loss_weights["ic4"] = base_w4 * w4

            # keep consistency gentle early on (ramp up over first ~30%)
            progress = min(1.0, global_step / max(1, int(0.3 * total_train_steps)))
            model.loss_weights["consistency"] = float(config["loss_weights"].get("consistency", 0.2)) * progress

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp, **amp_kwargs):
                out = model(**batch)
                loss = out["loss"]

            if precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if config.get("grad_clip", 0):
                if precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

                max_grad = max((p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), default=0)
                print(f"[Step {global_step}] Max grad: {max_grad}")

            if precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            for k, v in out["losses"].items():
                running[f"loss_{k}"] += float(v.item())
                counts[f"loss_{k}"] += 1

            correct, total = mlm_accuracy(out["mlm_logits"], batch["mlm_labels"])
            running["acc_mlm_correct"] += correct
            running["acc_mlm_total"] += total

            correct, total = binary_accuracy(out["sop_logits"], batch["sop_labels"])
            running["acc_sop_correct"] += correct
            running["acc_sop_total"] += total

            if out["ic2_logits"] is not None:
                c, t = top1_accuracy(out["ic2_logits"], batch["sic2"])
                running["acc_ic2_correct"] += c
                running["acc_ic2_total"] += t
            if out["ic3_logits"] is not None:
                c, t = top1_accuracy(out["ic3_logits"], batch["sic3"])
                running["acc_ic3_correct"] += c
                running["acc_ic3_total"] += t
            if out["ic4_logits"] is not None:
                c, t = top1_accuracy(out["ic4_logits"], batch["sic4"])
                running["acc_ic4_correct"] += c
                running["acc_ic4_total"] += t

            global_step += 1

            if step % max(1, config["logging_steps"] // 5) == 0 or step == 1:
                progress_bar.set_postfix({
                    "loss_mlm": f"{(running['loss_mlm'] / max(1, counts['loss_mlm'])):.3f}" if counts.get('loss_mlm', 0) else "-",
                    "loss_sop": f"{(running['loss_sop'] / max(1, counts['loss_sop'])):.3f}" if counts.get('loss_sop', 0) else "-",
                    "loss_ic2": f"{(running['loss_ic2'] / max(1, counts['loss_ic2'])):.3f}" if counts.get('loss_ic2', 0) else "-",
                    "loss_ic3": f"{(running['loss_ic3'] / max(1, counts['loss_ic3'])):.3f}" if counts.get('loss_ic3', 0) else "-",
                    "loss_ic4": f"{(running['loss_ic4'] / max(1, counts['loss_ic4'])):.3f}" if counts.get('loss_ic4', 0) else "-",
                    "cons": f"{(running['loss_consistency'] / max(1, counts['loss_consistency'])):.3f}" if counts.get('loss_consistency', 0) else "-",
                    "acc_sop": f"{(running['acc_sop_correct'] / max(1, running['acc_sop_total'])):.3f}" if running.get('acc_sop_total', 0) else "-",
                })

            if global_step % config["logging_steps"] == 0:
                msg = [f"epoch {epoch} step {global_step}"]
                for key in ["mlm", "sop", "ic2", "ic3", "ic4", "consistency"]:
                    loss_key = f"loss_{key}"
                    if counts.get(loss_key, 0):
                        msg.append(f"{loss_key}:{running[loss_key] / counts[loss_key]:.4f}")
                if running.get("acc_mlm_total", 0) > 0:
                    msg.append(f"acc_mlm:{running['acc_mlm_correct'] / max(1, running['acc_mlm_total']):.4f}")
                if running.get("acc_sop_total", 0) > 0:
                    msg.append(f"acc_sop:{running['acc_sop_correct'] / max(1, running['acc_sop_total']):.4f}")
                if running.get("acc_ic2_total", 0) > 0:
                    msg.append(f"acc_ic2:{running['acc_ic2_correct'] / running['acc_ic2_total']:.4f}")
                if running.get("acc_ic3_total", 0) > 0:
                    msg.append(f"acc_ic3:{running['acc_ic3_correct'] / running['acc_ic3_total']:.4f}")
                if running.get("acc_ic4_total", 0) > 0:
                    msg.append(f"acc_ic4:{running['acc_ic4_correct'] / running['acc_ic4_total']:.4f}")
                msg.append(
                    f"w2:{model.loss_weights['ic2']:.3f} w3:{model.loss_weights['ic3']:.3f} w4:{model.loss_weights['ic4']:.3f} wC:{model.loss_weights['consistency']:.3f}")
                print(" | ".join(msg))

            progress_bar.update(1)

        progress_bar.close()
        dt = time.time() - t0
        print(f"Epoch {epoch} finished in {dt / 60:.1f} min")
        run_eval(model, device, val_loader, desc=f"VAL epoch {epoch}")

    model.save_pretrained(config["save_dir"], safe_serialization=bool(config.get("safe_serialization", False)))
    tokenizer.save_pretrained(config["save_dir"])
    print(f"Saved to {config['save_dir']}")


if __name__ == "__main__":
    main()