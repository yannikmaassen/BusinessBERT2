import argparse
import os

def parse_cli_args():
    parser = argparse.ArgumentParser(description="BusinessBERT 2.0 – Pretraining")
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument("--data", required=False, help="Optional: override jsonl path")
    parser.add_argument("--report_to", required=False, help="Optional: override report_to in config")
    parser.add_argument("--batch_size", required=False, help="Optional: override batch_size")
    parser.add_argument("--max_seq_length", required=False, help="Optional: override max_seq_length")
    parser.add_argument("--max_steps", required=False, help="Optional: override max_steps")
    parser.add_argument("--learning_rate", required=False, help="Optional: override learning_rate")
    parser.add_argument("--num_train_epochs", required=False, help="Optional: override num_train_epochs")
    parser.add_argument("--warmup_steps", required=False, help="Optional: override warmup_steps")
    parser.add_argument("--weight_decay", required=False, help="Optional: override weight_decay")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], help="Precision for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to keep")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True, help="Whether to load the best model at end")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Which metric to use for best model")
    parser.add_argument("--greater_is_better", type=bool, default=False, help="Whether a greater metric is better")
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

    if args.batch_size is not None:
        args.batch_size = int(args.batch_size)

    if args.max_seq_length is not None:
        args.max_seq_length = int(args.max_seq_length)

    if args.max_steps is not None and args.max_steps != "none":
        args.max_steps = int(args.max_steps)

    if args.learning_rate is not None:
        args.learning_rate = float(args.learning_rate)

    if args.num_train_epochs is not None:
        args.num_train_epochs = int(args.num_train_epochs)

    if args.warmup_steps is not None:
        args.warmup_steps = int(args.warmup_steps)

    if args.weight_decay is not None:
        args.weight_decay = float(args.weight_decay)

    if args.precision == "fp16":
        args.precision = "fp16"
    elif args.precision == "bf16":
        args.precision = "bf16"
    else:
        args.precision = "fp32"

    if args.gradient_accumulation_steps is not None:
        args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)

    if args.num_workers is not None:
        args.num_workers = int(args.num_workers)

    if args.save_total_limit is not None:
        args.save_total_limit = int(args.save_total_limit)

    if args.load_best_model_at_end is not None:
        args.load_best_model_at_end = bool(args.load_best_model_at_end)

    if args.metric_for_best_model is not None:
        args.metric_for_best_model = str(args.metric_for_best_model)

    if args.greater_is_better is not None:
        args.greater_is_better = bool(args.greater_is_better)

    return args