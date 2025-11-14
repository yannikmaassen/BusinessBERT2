from transformers.trainer_utils import EvalLoopOutput
from transformers import Trainer
import wandb
import torch


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, taxonomy_maps=None, total_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxonomy_maps = taxonomy_maps
        self.total_steps = total_steps
        self.current_step = 0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to log individual losses and metrics to wandb.
        """
        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Log individual losses and metrics to WandB during training
        if model.training and self.state.global_step > 0 and self.args.report_to and "wandb" in self.args.report_to:
            # Extract individual loss components and metrics
            loss_dict = outputs.get("losses", {})
            metrics_dict = outputs.get("metrics", {})
            log_dict = {"train/total_loss": loss.item()}

            # Add individual losses
            for key, value in loss_dict.items():
                log_dict[f"train/loss_{key}"] = value.item()

            # Add metrics (accuracies)
            for key, value in metrics_dict.items():
                log_dict[f"train/{key}"] = value.item()

            # Log to WandB
            wandb.log(log_dict, step=self.state.global_step)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation loop to aggregate metrics"""
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Initialize accumulators with counters for each loss
        total_loss = 0.0
        total_losses = {}
        total_metrics = {}
        loss_counts = {}
        num_batches = 0

        # Iterate through evaluation batches
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Accumulate main loss (only if valid)
            if outputs.get("loss") is not None and not torch.isnan(outputs["loss"]):
                total_loss += outputs["loss"].item()

            # Accumulate individual losses (skip nan values)
            for key, value in outputs.get("losses", {}).items():
                if not torch.isnan(value):
                    if key not in total_losses:
                        total_losses[key] = 0.0
                        loss_counts[key] = 0
                    total_losses[key] += value.item()
                    loss_counts[key] += 1

            # Accumulate metrics
            for key, value in outputs.get("metrics", {}).items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value.item()

            num_batches += 1

        # Calculate averages
        metrics = {}
        metrics[f"{metric_key_prefix}_loss"] = total_loss / num_batches if num_batches > 0 else float('nan')

        # Use individual counts for each loss type
        for key, value in total_losses.items():
            count = loss_counts.get(key, 0)
            metrics[f"{metric_key_prefix}_loss_{key}"] = value / count if count > 0 else float('nan')

        for key, value in total_metrics.items():
            metrics[f"{metric_key_prefix}_{key}"] = value / num_batches if num_batches > 0 else float('nan')

        # Log to WandB (skip nan values)
        if self.args.report_to and "wandb" in self.args.report_to:
            log_dict = {}

            if not torch.isnan(torch.tensor(metrics[f"{metric_key_prefix}_loss"])):
                log_dict[f"{metric_key_prefix}/total_loss"] = metrics[f"{metric_key_prefix}_loss"]

            for key in total_losses.keys():
                metric_value = metrics[f"{metric_key_prefix}_loss_{key}"]
                if not torch.isnan(torch.tensor(metric_value)):
                    log_dict[f"{metric_key_prefix}/loss_{key}"] = metric_value

            for key in total_metrics.keys():
                log_dict[f"{metric_key_prefix}/{key}"] = metrics[f"{metric_key_prefix}_{key}"]

            if log_dict:
                wandb.log(log_dict, step=self.state.global_step, commit=False)

        # Return EvalLoopOutput for compatibility
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset)
        )