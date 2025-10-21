import os
import random
import argparse
import math
from typing import Dict
import torch
import yaml
import wandb
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertConfig,
    TrainingArguments,
    Trainer,
)
from src.utils.file_manager import read_jsonl
from src.utils.taxonomy import build_taxonomy_maps
from src.data import make_examples, PretrainDataset
from src.data.collator import BusinessBERTDataCollator
from src.models import BusinessBERT2Pretrain

# Set PyTorch memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, data_override: str = None) -> Dict:
    """Load YAML configuration with optional data path override"""
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    if data_override:
        config["jsonl_path"] = data_override
    return config


def coarse_to_fine_weights(training_step: int, total_steps: int):
    """Calculate weights for different SIC levels based on training progress"""
    progress = min(max(training_step / max(1, total_steps), 0.0), 1.0)
    cosine_ramp = 0.5 - 0.5 * math.cos(math.pi * progress)

    weight_sic2 = 0.8 * (1 - cosine_ramp) + 0.2   # 1.0 -> 0.2
    weight_sic3 = 0.3 + 0.4 * cosine_ramp         # 0.3 -> 0.7
    weight_sic4 = 0.1 + 0.9 * cosine_ramp         # 0.1 -> 1.0
    weight_sum = weight_sic2 + weight_sic3 + weight_sic4

    return {
        "ic2": weight_sic2 / weight_sum,
        "ic3": weight_sic3 / weight_sum,
        "ic4": weight_sic4 / weight_sum
    }


class CustomTrainer(Trainer):
    """Custom Trainer to update loss weights during training and log metrics to W&B"""
    def __init__(self, *args, **kwargs):
        self.base_loss_weights = kwargs.pop("loss_weights", {})
        super().__init__(*args, **kwargs)
        self.current_step = 0
        self.use_wandb = wandb is not None and wandb.run is not None

    def training_step(self, model, inputs):
        # Update the loss weights based on training progress
        step_weights = coarse_to_fine_weights(
            self.current_step,
            self.args.max_steps or self.args.num_train_epochs * self.steps_per_epoch
        )

        # Apply base weights and scaling factors
        model.loss_weights = {
            "mlm": self.base_loss_weights.get("mlm", 1.0),
            "sop": self.base_loss_weights.get("sop", 1.0),
            "ic2": self.base_loss_weights.get("ic2", 0.0) * step_weights["ic2"],
            "ic3": self.base_loss_weights.get("ic3", 0.0) * step_weights["ic3"],
            "ic4": self.base_loss_weights.get("ic4", 0.0) * step_weights["ic4"],
            # Ramp up consistency over first ~30% of training
            "consistency": self.base_loss_weights.get("consistency", 0.0) * min(1.0, self.current_step / (0.3 * (self.args.max_steps or self.args.num_train_epochs * self.steps_per_epoch)))
        }

        self.current_step += 1

        # Call the regular training step
        outputs = super().training_step(model, inputs)

        # Log detailed metrics to W&B
        if self.use_wandb and self.current_step % self.args.logging_steps == 0:
            self._log_training_metrics(model, inputs, outputs)

        return outputs

    def _log_training_metrics(self, model, inputs, outputs):
        """Log detailed training metrics to W&B"""
        try:
            # Extract detailed losses and metrics
            with torch.no_grad():
                # Get model outputs with all losses
                model_outputs = model(**inputs)

                # Base metrics
                metrics = {
                    "train/loss": outputs.loss.item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "global_step": self.current_step,
                }

                # Individual loss components
                if isinstance(model_outputs, dict) and "losses" in model_outputs:
                    for loss_name, loss_value in model_outputs["losses"].items():
                        metrics[f"train/loss_{loss_name}"] = loss_value.item()

                # Calculate accuracies
                if "mlm_logits" in model_outputs and "labels" in inputs:
                    pred = model_outputs["mlm_logits"].argmax(dim=-1)
                    mask = inputs["labels"] != -100
                    correct = (pred[mask] == inputs["labels"][mask]).sum()
                    total = mask.sum()
                    if total > 0:
                        metrics["train/mlm_accuracy"] = (correct / total).item()

                if "seq_relationship_logits" in model_outputs and "next_sentence_label" in inputs:
                    pred = model_outputs["seq_relationship_logits"].argmax(dim=-1)
                    correct = (pred == inputs["next_sentence_label"]).sum()
                    total = inputs["next_sentence_label"].shape[0]
                    metrics["train/sop_accuracy"] = (correct / total).item()

                # SIC classification accuracies
                for level in ["ic2", "ic3", "ic4"]:
                    logit_key = f"{level}_logits"
                    label_key = level.replace("ic", "sic")
                    if logit_key in model_outputs and label_key in inputs:
                        pred = model_outputs[logit_key].argmax(dim=-1)
                        mask = inputs[label_key] != -100
                        correct = (pred[mask] == inputs[label_key][mask]).sum()
                        total = mask.sum()
                        if total > 0:
                            metrics[f"train/{level}_accuracy"] = (correct / total).item()

                # Loss weights
                for key, value in model.loss_weights.items():
                    metrics[f"train/weight_{key}"] = value

                # Log metrics to W&B
                wandb.log(metrics, step=self.current_step)

        except Exception as e:
            # If there's an error during metrics logging, just print a warning and continue
            print(f"Warning: Error logging metrics to W&B: {e}")

    def evaluate(self, *args, **kwargs):
        """Override evaluate to add custom metrics logging for evaluation"""
        output = super().evaluate(*args, **kwargs)

        if self.use_wandb:
            # Log evaluation metrics to W&B
            metrics = {f"eval/{k}": v for k, v in output.items()}
            metrics["global_step"] = self.current_step
            wandb.log(metrics, step=self.current_step)

        return output

    @property
    def steps_per_epoch(self):
        """Calculate steps per epoch based on batch size and dataset size"""
        if not hasattr(self, "_steps_per_epoch"):
            self._steps_per_epoch = len(self.train_dataset) // (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            )
        return self._steps_per_epoch


def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BusinessBERT2 pretraining with simplified code")
    parser.add_argument("--config", type=str, default="config/pretrain.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Override data path from config")
    return parser.parse_args()


def main():
    # Disable tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_cli_args()

    # Load configuration
    config = load_config(args.config, args.data)
    output_dir = config["save_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Setup Weights & Biases if requested
    use_wandb = config.get("report_to", "").lower() == "wandb"
    if use_wandb and wandb is not None:
        wandb_kwargs = {
            "project": config.get("wandb_project", "businessbert2"),
            "mode": config.get("wandb_mode", "online"),
            "resume": "allow",
        }
        wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None}, config=config)

    # Load and prepare data
    print(f"Loading data from {config['jsonl_path']}")
    dataset = read_jsonl(config["jsonl_path"])
    print(f"Loaded {len(dataset)} rows")

    # Build taxonomy maps for industry classification
    taxonomy_maps = build_taxonomy_maps(
        dataset,
        config["field_sic2"],
        config["field_sic3"],
        config["field_sic4"]
    )
    print(
        f"SIC sizes: "
        f"2-digit={len(taxonomy_maps['sic2_list'])}, "
        f"3-digit={len(taxonomy_maps['sic3_list'])}, "
        f"4-digit={len(taxonomy_maps['sic4_list'])}"
    )

    # Split into training and validation sets
    train_rows, val_rows = train_test_split(
        dataset,
        test_size=config["val_ratio"],
        shuffle=True,
        random_state=config.get("seed", 42),
    )

    # Free up memory from the original dataset
    del dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create examples with sentence pairs and labels
    train_examples = make_examples(
        train_rows,
        config["field_text"],
        config["field_sic2"],
        config["field_sic3"],
        config["field_sic4"]
    )
    val_examples = make_examples(
        val_rows,
        config["field_text"],
        config["field_sic2"],
        config["field_sic3"],
        config["field_sic4"]
    )

    # Free up more memory
    del train_rows, val_rows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else "[PAD]"

    # Create datasets
    train_dataset = PretrainDataset(
        train_examples,
        tokenizer,
        config["max_seq_len"],
        taxonomy_maps["idx2"],
        taxonomy_maps["idx3"],
        taxonomy_maps["idx4"]
    )
    val_dataset = PretrainDataset(
        val_examples,
        tokenizer,
        config["max_seq_len"],
        taxonomy_maps["idx2"],
        taxonomy_maps["idx3"],
        taxonomy_maps["idx4"]
    )

    # Free up more memory
    del train_examples, val_examples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create data collator
    data_collator = BusinessBERTDataCollator(
        tokenizer=tokenizer,
        mlm_probability=config.get("mlm_probability", 0.15)
    )

    # Set up BERT configuration
    bert_config = BertConfig.from_pretrained(config["base_tokenizer"])

    # Print BERT configuration details
    print("\n===== BERT Configuration =====")
    print(f"Vocabulary size: {bert_config.vocab_size}")
    print(f"Hidden size: {bert_config.hidden_size}")
    print(f"Number of layers: {bert_config.num_hidden_layers}")
    print(f"Number of attention heads: {bert_config.num_attention_heads}")
    print(f"Intermediate size: {bert_config.intermediate_size}")
    print(f"Hidden dropout prob: {bert_config.hidden_dropout_prob}")
    print(f"Attention dropout prob: {bert_config.attention_probs_dropout_prob}")
    print(f"Max position embeddings: {bert_config.max_position_embeddings}")
    print(f"Type vocab size: {bert_config.type_vocab_size}")
    print("===============================\n")

    # Check if batch size needs to be reduced due to memory constraints
    original_batch_size = config.get("train_batch_size", 8)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024 * 1024 * 1024:  # If less than 10GB
        # Reduce batch size for smaller GPUs
        reduced_batch_size = max(1, original_batch_size // 2)
        print(f"WARNING: Reducing batch size from {original_batch_size} to {reduced_batch_size} to fit in GPU memory")
        config["train_batch_size"] = reduced_batch_size
        config["val_batch_size"] = max(1, config.get("val_batch_size", 8) // 2)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move taxonomy matrices to CPU first, only move to GPU when needed
    A32 = taxonomy_maps["A32"] if len(taxonomy_maps["sic3_list"]) and len(taxonomy_maps["sic2_list"]) else None
    A43 = taxonomy_maps["A43"] if len(taxonomy_maps["sic4_list"]) and len(taxonomy_maps["sic3_list"]) else None

    # Build model
    model = BusinessBERT2Pretrain(
        config=bert_config,
        n_sic2_classes=len(taxonomy_maps["sic2_list"]),
        n_sic3_classes=len(taxonomy_maps["sic3_list"]),
        n_sic4_classes=len(taxonomy_maps["sic4_list"]),
        A32=A32,
        A43=A43,
        loss_weights=config["loss_weights"],
    )

    # Clear memory after model initialization
    del taxonomy_maps, A32, A43
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define training arguments with memory optimization settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["val_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["num_warmup_steps"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config["logging_steps"],
        evaluation_strategy="steps" if config.get("eval_steps") else "no",
        eval_steps=config.get("eval_steps"),
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 3),
        # Memory optimization settings
        fp16=config.get("precision", "fp32").lower() == "fp16",
        bf16=config.get("precision", "fp32").lower() == "bf16",
        dataloader_num_workers=config["num_workers"],
        report_to="wandb" if use_wandb else "none",
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        # Enable gradient checkpointing to save memory
        gradient_checkpointing=True,
        max_grad_norm=config.get("grad_clip", 0) if config.get("grad_clip", 0) > 0 else None,
        # Add more memory optimization options
        optim="adamw_torch",
    )

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_weights=config["loss_weights"],
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Close wandb
    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
