import yaml
from typing import Dict


def load_config(args) -> Dict:
    """Load config from YAML (args.config) and override with CLI arguments in args.

    This preserves the previous behavior: any non-None CLI argument will override
    the corresponding value from the YAML file. For max_steps, the string
    'none' explicitly means: do not override the YAML value.
    """
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Override config values with CLI arguments if provided
    if getattr(args, "data", None):
        config["jsonl_path"] = args.data
    if getattr(args, "report_to", None):
        config["report_to"] = args.report_to
    if getattr(args, "max_seq_length", None) is not None:
        config["max_seq_len"] = args.max_seq_length
    if getattr(args, "batch_size", None) is not None:
        config["train_batch_size"] = args.batch_size
        config["val_batch_size"] = args.batch_size
    if getattr(args, "learning_rate", None) is not None:
        config["learning_rate"] = args.learning_rate
    if getattr(args, "num_train_epochs", None) is not None:
        config["num_train_epochs"] = args.num_train_epochs
    # Special-case: respect 'none' sentinel for max_steps.
    if getattr(args, "max_steps", None) is not None and getattr(args, "max_steps") != "none":
        config["max_steps"] = args.max_steps
    if getattr(args, "warmup_steps", None) is not None:
        config["num_warmup_steps"] = args.warmup_steps
    if getattr(args, "num_workers", None) is not None:
        config["num_workers"] = args.num_workers
    if getattr(args, "weight_decay", None) is not None:
        config["weight_decay"] = args.weight_decay
    if getattr(args, "gradient_accumulation_steps", None) is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if getattr(args, "grad_clip", None) is not None:
        config["grad_clip"] = args.grad_clip
    if getattr(args, "precision", None) is not None:
        config["precision"] = args.precision
    if getattr(args, "logging_steps", None) is not None:
        config["logging_steps"] = args.logging_steps
    if getattr(args, "eval_strategy", None) is not None:
        config["eval_strategy"] = args.eval_strategy
    if getattr(args, "eval_steps", None) is not None:
        config["eval_steps"] = args.eval_steps
    if getattr(args, "save_total_limit", None) is not None:
        config["save_total_limit"] = args.save_total_limit
    if getattr(args, "save_strategy", None) is not None:
        config["save_strategy"] = args.save_strategy
    if getattr(args, "save_steps", None) is not None:
        config["save_steps"] = args.save_steps
    if getattr(args, "load_best_model_at_end", None) is not None:
        config["load_best_model_at_end"] = args.load_best_model_at_end
    if getattr(args, "metric_for_best_model", None) is not None:
        config["metric_for_best_model"] = args.metric_for_best_model
    if getattr(args, "greater_is_better", None) is not None:
        config["greater_is_better"] = args.greater_is_better
    if getattr(args, "val_ratio", None) is not None:
        config["val_ratio"] = args.val_ratio
    if getattr(args, "save_dir", None):
        config["save_dir"] = args.save_dir
    if getattr(args, "save_safetensors", None) is not None:
        config["save_safetensors"] = args.save_safetensors
    if getattr(args, "safe_serialization", None) is not None:
        config["safe_serialization"] = args.safe_serialization
    if getattr(args, "wandb_mode", None) is not None:
        config["wandb_mode"] = args.wandb_mode
    if getattr(args, "wandb_project", None) is not None:
        config["wandb_project"] = args.wandb_project
    if getattr(args, "loss_weights", None) is not None:
        config["loss_weights"] = args.loss_weights

    return config
