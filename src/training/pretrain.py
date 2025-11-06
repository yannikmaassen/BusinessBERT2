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
from src.data import PretrainDataset, PretrainDatasetRandomSampling, PretrainDatasetOnTheFly, Collator
from src.models import BusinessBERT2Pretrain
from src.utils.arg_parser import parse_cli_args
from src.utils.taxonomy import build_taxonomy_maps


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(
    path: str,
    data_override: str = None,
    report_to: str = None,
    max_seq_length: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_train_epochs: int = None,
    max_steps: int = None,
    warmup_steps: int = None,
    num_workers: int = None,
    weight_decay: float = None,
    gradient_accumulation_steps: int = None,
    grad_clip: float = None,
    precision: str = None,
    logging_steps: int = None,
    eval_strategy: str = None,
    eval_steps: int = None,
    save_total_limit: int = None,
    save_strategy: str = None,
    save_steps: int = None,
    load_best_model_at_end: bool = None,
    metric_for_best_model: str = None,
    greater_is_better: bool = None,
    val_ratio: float = None,

) -> Dict:
    """Load config from YAML and override with CLI arguments if provided."""
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    # Override config values with CLI arguments if provided
    if data_override:
        config["jsonl_path"] = data_override
    if report_to:
        config["report_to"] = report_to
    if max_seq_length is not None:
        config["max_seq_len"] = max_seq_length
    if batch_size is not None:
        config["train_batch_size"] = batch_size
        config["val_batch_size"] = batch_size
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        config["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        config["max_steps"] = max_steps
    if warmup_steps is not None:
        config["num_warmup_steps"] = warmup_steps
    if num_workers is not None:
        config["num_workers"] = num_workers
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = gradient_accumulation_steps
    if grad_clip is not None:
        config["grad_clip"] = grad_clip
    if precision is not None:
        config["precision"] = precision
    if logging_steps is not None:
        config["logging_steps"] = logging_steps
    if eval_strategy is not None:
        config["eval_strategy"] = eval_strategy
    if eval_steps is not None:
        config["eval_steps"] = eval_steps
    if save_total_limit is not None:
        config["save_total_limit"] = save_total_limit
    if save_strategy is not None:
        config["save_strategy"] = save_strategy
    if save_steps is not None:
        config["save_steps"] = save_steps
    if load_best_model_at_end is not None:
        config["load_best_model_at_end"] = load_best_model_at_end
    if metric_for_best_model is not None:
        config["metric_for_best_model"] = metric_for_best_model
    if greater_is_better is not None:
        config["greater_is_better"] = greater_is_better
    if val_ratio is not None:
        config["val_ratio"] = val_ratio

    return config


def setup_tokenizer(base_tokenizer: str):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_cli_args()

    config = load_config(
        path=args.config,
        data_override=args.data,
        report_to=args.report_to,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        precision=args.precision,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        val_ratio=args.val_ratio,
    )
    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(int(config.get("seed", 42)))
    tokenizer = setup_tokenizer(config["base_tokenizer"])

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
    print("Loading dataset...")
    dataset = read_jsonl(config["jsonl_path"])
    print(f"Loaded {len(dataset)} rows")

    print("Splitting train/validation...")
    train_rows, val_rows = train_test_split(
        dataset,
        test_size=config["val_ratio"],
        shuffle=True,
        random_state=config.get("seed", 42),
    )

    taxonomy_maps = build_taxonomy_maps(dataset, config["field_sic2"], config["field_sic3"], config["field_sic4"])
    print(
        f"SIC sizes: "
        f"2-digit={len(taxonomy_maps['sic2_list'])}, "
        f"3-digit={len(taxonomy_maps['sic3_list'])}, "
        f"4-digit={len(taxonomy_maps['sic4_list'])}"
    )

    print("Tokenizing train/val datasets...")
    train_dataset = PretrainDatasetOnTheFly(train_rows, tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])
    val_dataset   = PretrainDatasetOnTheFly(val_rows,   tokenizer, config["max_seq_len"], taxonomy_maps["idx2"], taxonomy_maps["idx3"], taxonomy_maps["idx4"])

    data_collator = Collator(tokenizer=tokenizer)

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

    total_steps = config["max_steps"]

    training_args = TrainingArguments(
        output_dir=config["save_dir"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["val_batch_size"],
        warmup_steps=config["num_warmup_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        fp16=(config.get("precision") == "fp16"),
        bf16=(config.get("precision") == "bf16"),
        dataloader_num_workers=config["num_workers"],
        report_to=config.get("report_to", "none"),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["grad_clip"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config.get("load_best_model_at_end"),
        metric_for_best_model=config.get("metric_for_best_model"),
        greater_is_better=config.get("greater_is_better"),
        save_safetensors=False,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        taxonomy_maps=taxonomy_maps,
        total_steps=total_steps,
    )

    trainer.train(resume_from_checkpoint=True)

    model.save_pretrained(config["save_dir"], safe_serialization=bool(config.get("safe_serialization", False)))
    tokenizer.save_pretrained(config["save_dir"])
    print(f"Saved to {config['save_dir']}")


if __name__ == "__main__":
    main()