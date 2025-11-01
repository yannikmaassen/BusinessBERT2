from transformers import Trainer
import wandb
import math


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

            # Prepare logging dict
            log_dict = {"train/total_loss": loss.item()}

            # Add individual losses
            for key, value in loss_dict.items():
                log_dict[f"train/loss_{key}"] = value.item()

            # Add metrics (accuracies)
            for key, value in metrics_dict.items():
                log_dict[f"train/{key}"] = value.item()

            # Add current loss weights
            if hasattr(model, 'loss_weights'):
                log_dict["train/w_ic2"] = float(model.loss_weights.get("ic2", 0))
                log_dict["train/w_ic3"] = float(model.loss_weights.get("ic3", 0))
                log_dict["train/w_ic4"] = float(model.loss_weights.get("ic4", 0))
                log_dict["train/w_consistency"] = float(model.loss_weights.get("consistency", 0))

            # Log to WandB
            wandb.log(log_dict, step=self.state.global_step)

        # Store outputs for evaluation logging
        if not model.training:
            self._last_outputs = outputs

        return (loss, outputs) if return_outputs else loss

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     """
    #     Override to handle multi-task outputs and dynamic loss weights.
    #     """
    #     if model.training:
    #         # Log statistics every 100 steps
    #         if self.current_step > 0 and self.current_step % 100 == 0:
    #             total = self.sic_stats['total_samples']
    #             if total > 0:
    #                 print(f"\n=== SIC Code Statistics (Step {self.current_step}) ===", flush=True)
    #                 print(f"Total samples processed: {total}", flush=True)
    #
    #                 sic2_valid = total - self.sic_stats['sic2_invalid']
    #                 sic2_pct = (sic2_valid / total * 100)
    #                 print(f"SIC2 - Valid: {sic2_valid}/{total} ({sic2_pct:.2f}%) | Invalid: {self.sic_stats['sic2_invalid']}", flush=True)
    #
    #                 sic3_valid = total - self.sic_stats['sic3_invalid']
    #                 sic3_pct = (sic3_valid / total * 100)
    #                 print(f"SIC3 - Valid: {sic3_valid}/{total} ({sic3_pct:.2f}%) | Invalid: {self.sic_stats['sic3_invalid']}", flush=True)
    #
    #                 sic4_valid = total - self.sic_stats['sic4_invalid']
    #                 sic4_pct = (sic4_valid / total * 100)
    #                 print(f"SIC4 - Valid: {sic4_valid}/{total} ({sic4_pct:.2f}%) | Invalid: {self.sic_stats['sic4_invalid']}", flush=True)
    #                 print("=" * 50, flush=True)
    #
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

    @staticmethod
    def _coarse_to_fine_weights(step, total_steps):
        progress = min(max(step / max(1, total_steps), 0.0), 1.0)
        cosine_ramp = 0.5 - 0.5 * math.cos(math.pi * progress)

        w2 = 0.8 * (1 - cosine_ramp) + 0.2
        w3 = 0.3 + 0.4 * cosine_ramp
        w4 = 0.1 + 0.9 * cosine_ramp

        total = w2 + w3 + w4
        return w2 / total, w3 / total, w4 / total

    def log(self, logs, *args, **kwargs):
        # Add current loss weights (for both train and eval)
        if hasattr(self.model, 'loss_weights'):
            prefix = "eval_" if "eval_loss" in logs else ""
            logs[f"{prefix}w_ic2"] = float(self.model.loss_weights.get("ic2", 0))
            logs[f"{prefix}w_ic3"] = float(self.model.loss_weights.get("ic3", 0))
            logs[f"{prefix}w_ic4"] = float(self.model.loss_weights.get("ic4", 0))
            logs[f"{prefix}w_consistency"] = float(self.model.loss_weights.get("consistency", 0))

        # Handle evaluation outputs
        if hasattr(self, '_last_outputs') and self._last_outputs:
            outputs = self._last_outputs
            losses = outputs.get("losses", {})
            metrics = outputs.get("metrics", {})
            prefix = "eval_" if "eval_loss" in logs else ""

            # Individual losses
            for loss_key, loss_value in losses.items():
                logs[f"{prefix}loss_{loss_key}"] = float(loss_value)

            # Individual metrics (accuracies)
            for metric_key, metric_value in metrics.items():
                logs[f"{prefix}{metric_key}"] = float(metric_value)

        # Call parent's log with all arguments
        super().log(logs, *args, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        """Override to aggregate metrics across evaluation batches"""
        # Call parent evaluation
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        # Extract aggregated metrics from the last batch
        return output
