import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    val = str(v).strip().lower()
    if val in ("true", "True", "1"):
        return True
    if val in ("false", "False", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="BusinessBERT 2.0 – Pretraining")
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument("--data", required=False, help="Optional: override jsonl path")
    parser.add_argument("--report_to", required=False, help="Optional: override report_to in config")
    parser.add_argument("--max_seq_length", required=False, help="Optional: override max_seq_length")
    parser.add_argument("--batch_size", required=False, help="Optional: override batch_size")
    parser.add_argument("--learning_rate", required=False, help="Optional: override learning_rate")
    parser.add_argument("--num_train_epochs", required=False, help="Optional: override num_train_epochs")
    parser.add_argument("--max_steps", required=False, help="Optional: override max_steps")
    parser.add_argument("--warmup_steps", required=False, help="Optional: override warmup_steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--weight_decay", required=False, help="Optional: override weight_decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], help="Precision for training")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy: 'steps' or 'epoch'")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to keep")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy: 'steps' or 'epoch'")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True, help="Whether to load the best model at end")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Which metric to use for best model")
    parser.add_argument("--greater_is_better", type=bool, default=False, help="Whether a greater metric is better")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--save_dir", type=str, default="./outputs/pretrain", help="Directory to save the model and tokenizer")
    parser.add_argument("--save_safetensors", type=bool, default=False, help="Whether to save model in safetensors format")
    parser.add_argument("--safe_serialization", type=bool, default=False, help="Whether to use safe serialization when saving the model")
    parser.add_argument("--wandb_mode", type=str, default="online", help="WandB mode: 'online', 'offline', or 'disabled'")
    parser.add_argument("--wandb_project", type=str, default="pretraining-businessbert2", help="WandB project name")
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "..", "..", "config", "pretrain.yaml")
        print(f"No config path specified, using default: {args.config}")

    if args.data is None:
        args.data = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample.jsonl")
        print(f"No data path specified, using default: {args.data}")

    if args.report_to is None:
        args.report_to = "none"
        print(f"No report_to specified, using default: {args.report_to}")

    if args.max_seq_length is not None:
        args.max_seq_length = int(args.max_seq_length)

    if args.batch_size is not None:
        args.batch_size = int(args.batch_size)

    if args.learning_rate is not None:
        args.learning_rate = float(args.learning_rate)

    if args.num_train_epochs is not None:
        args.num_train_epochs = int(args.num_train_epochs)

    if args.max_steps is not None and args.max_steps != "none":
        args.max_steps = int(args.max_steps)

    if args.warmup_steps is not None:
        args.warmup_steps = int(args.warmup_steps)

    if args.num_workers is not None:
        args.num_workers = int(args.num_workers)

    if args.weight_decay is not None:
        args.weight_decay = float(args.weight_decay)

    if args.gradient_accumulation_steps is not None:
        args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)

    if args.grad_clip is not None:
        args.grad_clip = float(args.grad_clip)

    if args.precision == "fp16":
        args.precision = "fp16"
    elif args.precision == "bf16":
        args.precision = "bf16"
    else:
        args.precision = "fp32"

    if args.logging_steps is not None:
        args.logging_steps = int(args.logging_steps)

    if args.eval_steps is not None:
        args.eval_steps = int(args.eval_steps)

    if args.save_total_limit is not None:
        args.save_total_limit = int(args.save_total_limit)

    if args.save_steps is not None:
        args.save_steps = int(args.save_steps)

    print("############### DEBUG ARGS ###############")
    print(args.load_best_model_at_end)

    if args.load_best_model_at_end is not None:
        args.load_best_model_at_end = args.load_best_model_at_end
        print(args.load_best_model_at_end)

    if args.metric_for_best_model is not None:
        args.metric_for_best_model = str(args.metric_for_best_model)

    if args.greater_is_better is not None:
        args.greater_is_better = bool(args.greater_is_better)

    if args.val_ratio is not None:
        args.val_ratio = float(args.val_ratio)

    if args.save_dir is not None:
        args.save_dir = str(args.save_dir)

    if args.save_safetensors is not None:
        args.save_safetensors = bool(args.save_safetensors)

    if args.safe_serialization is not None:
        args.safe_serialization = bool(args.safe_serialization)

    if args.wandb_mode not in ["online", "offline", "disabled"]:
        args.wandb_mode = "online"

    if args.wandb_project is not None:
        args.wandb_project = str(args.wandb_project)

    return args