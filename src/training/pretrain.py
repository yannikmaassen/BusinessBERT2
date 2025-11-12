import os
import random
import wandb
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, BertConfig, TrainingArguments
from src.training.trainer import MultiTaskTrainer
from src.utils.file_manager import read_jsonl
from src.data import PretrainDatasetOnTheFly, Collator
from src.models import BusinessBERT2Pretrain
from src.utils.arg_parser import parse_cli_args
from src.utils.taxonomy import build_taxonomy_maps
from src.utils.config_loader import load_config
from src.utils.checkpoint_finder import find_latest_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_tokenizer(base_tokenizer: str):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def main():
    # force legacy torch.load behavior to load checkpoints
    _orig_load = torch.load
    def _load_force_weights_false(f, *args, **kwargs):
         kwargs.setdefault("weights_only", False)
         return _orig_load(f, *args, **kwargs)
    torch.load = _load_force_weights_false

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_cli_args()

    config = load_config(args)

    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    os.makedirs(config["save_dir"], exist_ok=True)

    set_seed(int(config.get("seed", 42)))
    tokenizer = setup_tokenizer(config["base_tokenizer"])

    use_wandb = (config.get("report_to", "").lower() == "wandb")
    wb = None
    if use_wandb and wandb is not None:
        wb_init_kwargs = dict(
            project=config["wandb_project"],
            mode=config["wandb_mode"],
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
        consistency_warmup_ratio=0.2,
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
        save_safetensors=config["save_safetensors"]
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

    last_checkpoint = find_latest_checkpoint(config["save_dir"])
    print(f"Found latest checkpoint at: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint is not None else False)

    model.save_pretrained(config["save_dir"], safe_serialization=config["safe_serialization"])
    tokenizer.save_pretrained(config["save_dir"])
    print(f"Saved to {config['save_dir']}")


if __name__ == "__main__":
    main()