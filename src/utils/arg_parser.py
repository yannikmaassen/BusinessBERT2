import argparse
import os

def parse_cli_args():
    parser = argparse.ArgumentParser(description="BusinessBERT 2.0 – Pretraining")
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument("--data", required=False, help="Optional: override jsonl path")
    parser.add_argument("--report_to", required=False, help="Optional: override report_to in config")
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

    return args