from typing import Dict, Optional, Tuple, Any
import collections

import torch
from transformers import Trainer


def _to_item(value: torch.Tensor) -> float:
    try:
        return float(value.detach().mean().item())
    except Exception:
        return float(value)


class MultiTaskTrainer(Trainer):
    """
    Custom Trainer that logs per-component losses and accuracies for:
      - losses: mlm, ic2, ic3, ic4, consistency
      - accuracies: mlm, ic2, ic3, ic4

    Logging conventions:
      - Training step keys are prefixed with 'train/'.
      - Evaluation aggregated keys are prefixed with 'eval/'.
      - Overall scalar loss is logged as 'train/loss' and 'eval/loss'.
    """
    def __init__(self, *args, **kwargs):
        # Pop custom args you pass from main()
        self.taxonomy_maps = kwargs.pop("taxonomy_maps", None)
        self.total_steps = kwargs.pop("total_steps", None)
        super().__init__(*args, **kwargs)

    # ------------------------
    # TRAINING: per-step logging
    # ------------------------
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Runs the model forward and:
          - returns the main loss for optimization,
          - logs per-task losses and accuracies to the Trainer logger (-> W&B).
        """
        inputs = self._prepare_inputs(inputs)

        # Use smart autocast context (handles fp16/bf16 automatically)
        with self.autocast_smart_context_manager():
            outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs

        # -------- Logging (train/...) --------
        if isinstance(outputs, dict):
            to_log = {"train/loss": _to_item(outputs.get("loss", loss))}
            # Per-loss components
            for name in ["mlm", "ic2", "ic3", "ic4", "consistency"]:
                if "losses" in outputs and isinstance(outputs["losses"], dict) and name in outputs["losses"]:
                    to_log[f"train/{name}_loss"] = _to_item(outputs["losses"][name])

            # Accuracies
            if "metrics" in outputs and isinstance(outputs["metrics"], dict):
                for name in ["mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
                    if name in outputs["metrics"]:
                        to_log[f"train/{name}"] = _to_item(outputs["metrics"][name])

            # Respect logging frequency if configured
            should_log_now = True
            if self.state.global_step is not None and self.args.logging_steps and self.args.logging_steps > 0:
                # compute_loss is called before global_step increments; this logs on the *previous* step boundary.
                should_log_now = (self.state.global_step % self.args.logging_steps == 0)

            if should_log_now:
                self.log(to_log)

        return (loss, outputs) if return_outputs else loss

    # ------------------------
    # EVALUATION: aggregate + log
    # ------------------------
    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Custom evaluate loop that averages per-batch metrics and logs them with 'eval/' prefix.
        Returns the metrics dict (so it shows up in Trainer prints and callbacks).
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        # Running sums (weighted by batch size where possible)
        sums = collections.defaultdict(float)
        counts = collections.defaultdict(float)

        total_examples = 0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            batch_size = None
            for v in inputs.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    batch_size = v.size(0)
                    break
            if batch_size is None:
                batch_size = 1

            with self.autocast_smart_context_manager():
                outputs = model(**inputs)

            # Main loss
            total_loss_val = outputs.get("loss", None) if isinstance(outputs, dict) else None
            if total_loss_val is not None:
                sums["loss"] += _to_item(total_loss_val) * batch_size
                counts["loss"] += batch_size

            # Component losses
            for name in ["mlm", "ic2", "ic3", "ic4", "consistency"]:
                if isinstance(outputs, dict) and "losses" in outputs and name in outputs["losses"]:
                    sums[f"{name}_loss"] += _to_item(outputs["losses"][name]) * batch_size
                    counts[f"{name}_loss"] += batch_size

            # Accuracies (note: we weight by batch size; for MLM this approximates token-weighting)
            if isinstance(outputs, dict) and "metrics" in outputs:
                for name in ["mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
                    if name in outputs["metrics"]:
                        sums[name] += _to_item(outputs["metrics"][name]) * batch_size
                        counts[name] += batch_size

            total_examples += batch_size

        # Build averaged metrics
        averaged: Dict[str, float] = {}
        def _finalize(key_in: str, key_out: str):
            if counts.get(key_in, 0) > 0:
                averaged[key_out] = sums[key_in] / counts[key_in]

        _finalize("loss", f"{metric_key_prefix}/loss")
        for name in ["mlm", "ic2", "ic3", "ic4", "consistency"]:
            _finalize(f"{name}_loss", f"{metric_key_prefix}/{name}_loss")
        for name in ["mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
            _finalize(name, f"{metric_key_prefix}/{name}")

        self.log(averaged)

        # Add the "eval/samples" for completeness
        averaged[f"{metric_key_prefix}/samples"] = float(total_examples)

        return averaged
