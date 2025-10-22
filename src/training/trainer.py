from transformers import Trainer
import torch


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, taxonomy_maps=None, total_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxonomy_maps = taxonomy_maps
        self.total_steps = total_steps
        self.current_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to handle multi-task outputs and dynamic loss weights.
        """
        # Update loss weights based on training progress
        if self.total_steps:
            w2, w3, w4 = self._coarse_to_fine_weights(self.current_step, self.total_steps)
            base_weights = model.loss_weights.copy()
            model.loss_weights["ic2"] = base_weights["ic2"] * w2
            model.loss_weights["ic3"] = base_weights["ic3"] * w3
            model.loss_weights["ic4"] = base_weights["ic4"] * w4

            # Ramp consistency
            progress = min(1.0, self.current_step / max(1, int(0.3 * self.total_steps)))
            model.loss_weights["consistency"] = base_weights.get("consistency", 0.2) * progress

        self.current_step += 1

        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _coarse_to_fine_weights(step, total_steps):
        """Same logic as your current implementation"""
        import math
        progress = min(max(step / max(1, total_steps), 0.0), 1.0)
        cosine_ramp = 0.5 - 0.5 * math.cos(math.pi * progress)

        w2 = 0.8 * (1 - cosine_ramp) + 0.2
        w3 = 0.3 + 0.4 * cosine_ramp
        w4 = 0.1 + 0.9 * cosine_ramp

        total = w2 + w3 + w4
        return w2 / total, w3 / total, w4 / total

    def log(self, logs, *args, **kwargs):
        """Add custom metrics to logging"""
        # Add current loss weights
        if hasattr(self.model, 'loss_weights'):
            logs["w_ic2"] = float(self.model.loss_weights.get("ic2", 0))
            logs["w_ic3"] = float(self.model.loss_weights.get("ic3", 0))
            logs["w_ic4"] = float(self.model.loss_weights.get("ic4", 0))
            logs["w_consistency"] = float(self.model.loss_weights.get("consistency", 0))

        # Call parent's log with all arguments
        super().log(logs, *args, **kwargs)

