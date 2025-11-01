from transformers import Trainer
import wandb


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

            # Log to WandB
            wandb.log(log_dict, step=self.state.global_step)

        # Store outputs for evaluation logging
        if not model.training:
            self._last_outputs = outputs

        return (loss, outputs) if return_outputs else loss


    def log(self, logs, *args, **kwargs):
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
