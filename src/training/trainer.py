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
        Override to handle multi-task outputs.
        """
        # # Update loss weights based on training progress
        # if self.total_steps:
        #     w2, w3, w4 = self._coarse_to_fine_weights(self.current_step, self.total_steps)
        #     base_weights = model.loss_weights.copy()
        #     model.loss_weights["ic2"] = base_weights["ic2"] * w2
        #     model.loss_weights["ic3"] = base_weights["ic3"] * w3
        #     model.loss_weights["ic4"] = base_weights["ic4"] * w4
        #
        #     # Ramp consistency
        #     progress = min(1.0, self.current_step / max(1, int(0.3 * self.total_steps)))
        #     model.loss_weights["consistency"] = base_weights.get("consistency", 0.2) * progress
        #
        # self.current_step += 1

        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Store individual losses and accuracies for logging
        if not self.model.training:  # During evaluation
            self._last_outputs = outputs

        # Log individual losses to WandB (only during training)
        if self.state.global_step > 0 and self.args.report_to and "wandb" in self.args.report_to:
            # Extract individual loss components
            loss_dict = outputs.get("losses", {})

            # Prepare logging dict with proper names
            log_dict = {"train/loss_total": loss.item()}
            for key, value in loss_dict.items():
                log_dict[f"train/loss_{key}"] = value.item()

            # Add global_step for proper x-axis alignment
            log_dict["global_step"] = self.state.global_step

            # Log to WandB
            wandb.log({
                **{f"loss/{k}": v.item() for k, v in outputs["losses"].items()},
                **{f"metrics/{k}": v.item() for k, v in outputs["metrics"].items()},
            })

        return (loss, outputs) if return_outputs else loss

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

    def log(self, logs, *args, **kwargs):
        """Override to log individual losses during evaluation"""
        if hasattr(self, '_last_outputs') and self._last_outputs:
            outputs = self._last_outputs
            losses = outputs["losses"]
            prefix = "eval_" if "eval_loss" in logs else ""

            # Individual losses
            if "mlm" in losses:
                logs[f"{prefix}loss_mlm"] = float(losses["mlm"])
            if "ic2" in losses:
                logs[f"{prefix}loss_ic2"] = float(losses["ic2"])
            if "ic3" in losses:
                logs[f"{prefix}loss_ic3"] = float(losses["ic3"])
            if "ic4" in losses:
                logs[f"{prefix}loss_ic4"] = float(losses["ic4"])
            if "consistency" in losses:
                logs[f"{prefix}loss_consistency"] = float(losses["consistency"])

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
