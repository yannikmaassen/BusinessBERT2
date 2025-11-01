from transformers import Trainer
import wandb
import math
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
        Metrics are calculated by the model, not here.
        """
        # Forward pass - model handles all metric calculation
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Log to WandB during training only
        if model.training and self.state.global_step > 0 and self.args.report_to and "wandb" in self.args.report_to:
            log_dict = {"train/total_loss": loss.item()}

            # Extract losses from model output
            loss_dict = outputs.get("losses", {})
            for key, value in loss_dict.items():
                log_dict[f"train/loss_{key}"] = value.item()

            # Extract metrics (accuracies) from model output
            metrics_dict = outputs.get("metrics", {})
            for key, value in metrics_dict.items():
                log_dict[f"train/{key}"] = value.item()

            wandb.log(log_dict, step=self.state.global_step)

        return (loss, outputs) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     """
    #     Override to handle multi-task outputs and dynamic loss weights.
    #     """
    #     # Update loss weights based on training progress
    #     if self.total_steps:
    #         w2, w3, w4 = self._coarse_to_fine_weights(self.current_step, self.total_steps)
    #         base_weights = model.loss_weights.copy()
    #         # TODO: base weights are adjusted here; consider passing them from outside instead
    #         model.loss_weights["ic2"] = base_weights["ic2"] * w2
    #         model.loss_weights["ic3"] = base_weights["ic3"] * w3
    #         model.loss_weights["ic4"] = base_weights["ic4"] * w4
    #
    #         # Ramp consistency
    #         progress = min(1.0, self.current_step / max(1, int(0.3 * self.total_steps)))
    #         model.loss_weights["consistency"] = base_weights.get("consistency", 0.2) * progress
    #
    #     self.current_step += 1
    #
    #     # Forward pass
    #     outputs = model(**inputs)
    #     loss = outputs["loss"]
    #
    #     # Store individual losses and accuracies for logging
    #     if not self.model.training:  # During evaluation
    #         self._last_outputs = outputs
    #
    #     # Log individual losses to WandB (only during training)
    #     if self.state.global_step > 0 and self.args.report_to and "wandb" in self.args.report_to:
    #         # Extract individual loss components
    #         loss_dict = outputs.get("losses", {})
    #
    #         # Prepare logging dict with proper names
    #         log_dict = {"train/loss_total": loss.item()}
    #         for key, value in loss_dict.items():
    #             log_dict[f"train/loss_{key}"] = value.item()
    #
    #         # Add global_step for proper x-axis alignment
    #         log_dict["global_step"] = self.state.global_step
    #
    #         # Log to WandB
    #         wandb.log({
    #             **{f"loss/{k}": v.item() for k, v in outputs["losses"].items()},
    #             **{f"metrics/{k}": v.item() for k, v in outputs["metrics"].items()},
    #         })
    #
    #     return (loss, outputs) if return_outputs else loss

    # @staticmethod
    # def _coarse_to_fine_weights(step, total_steps):
    #     progress = min(max(step / max(1, total_steps), 0.0), 1.0)
    #     cosine_ramp = 0.5 - 0.5 * math.cos(math.pi * progress)
    #
    #     w2 = 0.8 * (1 - cosine_ramp) + 0.2
    #     w3 = 0.3 + 0.4 * cosine_ramp
    #     w4 = 0.1 + 0.9 * cosine_ramp
    #
    #     total = w2 + w3 + w4
    #     return w2 / total, w3 / total, w4 / total

    # def log(self, logs, *args, **kwargs):
    #     # Add current loss weights (for both train and eval)
    #     prefix = "eval_" if "eval_loss" in logs else ""
    #
    #     # Handle evaluation outputs
    #     if hasattr(self, '_last_outputs') and self._last_outputs:
    #         outputs = self._last_outputs
    #         losses = outputs.get("losses", {})
    #         metrics = outputs.get("metrics", {})
    #
    #         # Individual losses
    #         for loss_key, loss_value in losses.items():
    #             logs[f"{prefix}loss_{loss_key}"] = float(loss_value)
    #
    #         # Individual metrics (accuracies)
    #         for metric_key, metric_value in metrics.items():
    #             logs[f"{prefix}{metric_key}"] = float(metric_value)
    #
    #     # Call parent's log with all arguments
    #     super().log(logs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        """Override to inject aggregated eval metrics"""
        # If we have aggregated eval metrics, add them to logs
        if hasattr(self, '_eval_metrics') and self._eval_metrics:
            logs.update(self._eval_metrics)
            del self._eval_metrics  # Clear after logging

        # Call parent's log
        super().log(logs, *args, **kwargs)

    # def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
    #     """Override to aggregate metrics across evaluation batches"""
    #     # Call parent evaluation
    #     output = super().evaluation_loop(
    #         dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
    #     )
    #
    #     # Extract aggregated metrics from the last batch
    #     return output

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        """Override to aggregate metrics across evaluation batches"""
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Storage for aggregating across batches
        aggregated_losses = []
        aggregated_metrics = {}

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                outputs = model(**inputs)

            # Collect total loss
            aggregated_losses.append(outputs["loss"].item())

            # Collect individual losses
            losses = outputs.get("losses", {})
            for key, value in losses.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value.item())

            # Collect metrics (accuracies) - model already calculated them
            metrics = outputs.get("metrics", {})
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value.item())

        # Average all collected metrics
        eval_logs = {
            f"{metric_key_prefix}/total_loss": sum(aggregated_losses) / len(aggregated_losses)
        }

        for key, values in aggregated_metrics.items():
            if values:
                # Losses get _loss suffix, accuracies already have _accuracy suffix from model
                suffix = "_loss" if not key.endswith("_accuracy") else ""
                eval_logs[f"{metric_key_prefix}/{key}{suffix}"] = sum(values) / len(values)

        # Store for logging
        self._eval_metrics = eval_logs

        # Call parent to get standard output format
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return output
