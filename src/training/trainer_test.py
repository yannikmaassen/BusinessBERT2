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
    def __init__(self, *args, **kwargs):
        self.taxonomy_maps = kwargs.pop("taxonomy_maps", None)
        self.total_steps = kwargs.pop("total_steps", None)
        super().__init__(*args, **kwargs)

    # ------------ TRAIN (per-step) ------------
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        inputs = self._prepare_inputs(inputs)
        with self.autocast_smart_context_manager():
            outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs

        # ❗️Do NOT prefix with 'train/' — grouping comes from step key 'train/global_step'
        if isinstance(outputs, dict):
            to_log = {"loss": _to_item(outputs.get("loss", loss))}

            # component losses
            comp_losses = outputs.get("losses", {}) or {}
            for name in ["mlm", "ic2", "ic3", "ic4", "consistency"]:
                if name in comp_losses:
                    to_log[f"{name}_loss"] = _to_item(comp_losses[name])

            # accuracies
            comp_metrics = outputs.get("metrics", {}) or {}
            for name in ["mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
                if name in comp_metrics:
                    to_log[name] = _to_item(comp_metrics[name])

            # respect logging_steps
            should_log_now = True
            if self.state.global_step is not None and self.args.logging_steps and self.args.logging_steps > 0:
                should_log_now = (self.state.global_step % self.args.logging_steps == 0)

            if should_log_now:
                self.log(to_log)

        return (loss, outputs) if return_outputs else loss

    # ------------ EVAL (aggregate + log) ------------
    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        sums, counts = collections.defaultdict(float), collections.defaultdict(float)
        total_examples = 0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            batch_size = next((v.size(0) for v in inputs.values()
                               if isinstance(v, torch.Tensor) and v.dim() > 0), 1)

            with self.autocast_smart_context_manager():
                outputs = model(**inputs)

            if isinstance(outputs, dict) and "loss" in outputs:
                sums["loss"] += _to_item(outputs["loss"]) * batch_size
                counts["loss"] += batch_size

            comp_losses = outputs.get("losses", {}) if isinstance(outputs, dict) else {}
            for name in ["mlm", "ic2", "ic3", "ic4", "consistency"]:
                if name in comp_losses:
                    sums[f"{name}_loss"] += _to_item(comp_losses[name]) * batch_size
                    counts[f"{name}_loss"] += batch_size

            comp_metrics = outputs.get("metrics", {}) if isinstance(outputs, dict) else {}
            for name in ["mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
                if name in comp_metrics:
                    sums[name] += _to_item(comp_metrics[name]) * batch_size
                    counts[name] += batch_size

            total_examples += batch_size

        def _avg(k: str) -> Optional[float]:
            return (sums[k] / counts[k]) if counts.get(k, 0) > 0 else None

        # For W&B logging → ❗️no 'eval/' prefix; step key is already 'eval/global_step'
        to_log = {}
        for k in ["loss",
                  "mlm_loss", "ic2_loss", "ic3_loss", "ic4_loss", "consistency_loss",
                  "mlm_accuracy", "ic2_accuracy", "ic3_accuracy", "ic4_accuracy"]:
            val = _avg(k)
            if val is not None:
                to_log[k] = val

        self.log(to_log)

        # For Trainer return/printouts → keep classic 'eval_*' names
        to_return: Dict[str, float] = {}
        mapping = {
            "loss": "eval_loss",
            "mlm_loss": "eval_mlm_loss",
            "ic2_loss": "eval_ic2_loss",
            "ic3_loss": "eval_ic3_loss",
            "ic4_loss": "eval_ic4_loss",
            "consistency_loss": "eval_consistency_loss",
            "mlm_accuracy": "eval_mlm_accuracy",
            "ic2_accuracy": "eval_ic2_accuracy",
            "ic3_accuracy": "eval_ic3_accuracy",
            "ic4_accuracy": "eval_ic4_accuracy",
        }
        for k, outk in mapping.items():
            if k in to_log:
                to_return[outk] = to_log[k]

        to_return["eval_samples"] = float(total_examples)
        return to_return
