import os
import random
from typing import Dict
import wandb

from sklearn.model_selection import train_test_split
import torch
import yaml

from transformers import AutoTokenizer, BertConfig, TrainingArguments

from src.training.trainer import MultiTaskTrainer
from src.utils.file_manager import read_jsonl
from src.data import make_examples, PretrainDataset, Collator
from src.models import BusinessBERT2Pretrain
from src.utils.arg_parser import parse_cli_args
from src.utils.taxonomy import build_taxonomy_maps


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, data_override: str = None, report_to: str = None) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    if data_override:
        config["jsonl_path"] = data_override
    if report_to:
        config["report_to"] = report_to
    return config


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_cli_args()

    config = load_config(args.config, args.data, args.report_to)
    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    use_wandb = (config.get("report_to", "").lower() == "wandb")
    wb = None
    if use_wandb and wandb is not None:
        wb_init_kwargs = dict(
            project=config.get("wandb_project", "businessbert2"),
            mode=config.get("wandb_mode", "online"),   # "online" | "offline" | "disabled"
            resume="allow",
        )
        wb_init_kwargs = {k: v for k, v in wb_init_kwargs.items() if v is not None}
        wb = wandb.init(**wb_init_kwargs, config=config)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    # ---------------- Data ----------------
    dataset = read_jsonl(config["jsonl_path"])
    print(f"Loaded {len(dataset)} rows")

    print("Splitting train/validation...")
    train_rows, val_rows = train_test_split(
        dataset,
        test_size=config["val_ratio"],
        shuffle=True,
        random_state=config.get("seed", 42),
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else "[PAD]"

    print("Building taxonomy maps...")
    taxonomy_maps = build_taxonomy_maps(dataset, config["field_sic2"], config["field_sic3"], config["field_sic4"])
    print(
        f"SIC sizes: "
        f"2-digit={len(taxonomy_maps['sic2_list'])}, "
        f"3-digit={len(taxonomy_maps['sic3_list'])}, "
        f"4-digit={len(taxonomy_maps['sic4_list'])}"
    )

    print("Tokenizing datasets...")
    train_dataset = PretrainDataset(train_rows, tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])
    val_dataset   = PretrainDataset(val_rows,   tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])

    collate = Collator(
        tokenizer=tokenizer
    )

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig.from_pretrained(config["base_tokenizer"])

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

    if use_wandb and wandb is not None and config.get("wandb_watch", True):
        wandb.watch(model, log="gradients", log_freq=max(1, int(config.get("logging_steps", 50))))

    # Calculate total steps
    steps_per_epoch = len(train_dataset) // config["train_batch_size"]
    total_steps = config["num_train_epochs"] * steps_per_epoch

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["save_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["val_batch_size"],
        warmup_steps=config["num_warmup_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=config.get("precision") == "fp16",
        bf16=config.get("precision") == "bf16",
        dataloader_num_workers=config["num_workers"],
        report_to="wandb" if config.get("report_to") == "wandb" else "none",
        gradient_accumulation_steps=1,
        max_grad_norm=config.get("grad_clip", 1.0),
        save_safetensors=False,
    )

    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
        taxonomy_maps=taxonomy_maps,
        total_steps=total_steps,
    )

    # Train
    trainer.train()

    model.save_pretrained(config["save_dir"], safe_serialization=bool(config.get("safe_serialization", False)))
    tokenizer.save_pretrained(config["save_dir"])
    print(f"Saved to {config['save_dir']}")


if __name__ == "__main__":
    main()