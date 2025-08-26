import collections
from typing import Dict
import torch
from src.training.metrics import mlm_accuracy, top1_accuracy, binary_accuracy


# TODO: IC
@torch.no_grad()
def run_eval(model, device, loader, desc: str = "VAL") -> Dict:
    model.eval()
    loss_sums = collections.defaultdict(float)
    counts = collections.defaultdict(int)

    mlm_correct = 0; mlm_total = 0
    sop_correct = 0; sop_total = 0
    ic2_correct = 0; ic2_total = 0
    ic3_correct = 0; ic3_total = 0
    ic4_correct = 0; ic4_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)

            for k, v in out["losses"].items():
                loss_sums[k] += v.item()
                counts[k] += 1

            c, t = mlm_accuracy(out["mlm_logits"], batch["mlm_labels"]); mlm_correct += c; mlm_total += t
            c, t = binary_accuracy(out["sop_logits"], batch["sop_labels"]); sop_correct += c; sop_total += t
            if out["ic2_logits"] is not None:
                c, t = top1_accuracy(out["ic2_logits"], batch["sic2"]); ic2_correct += c; ic2_total += t
            if out["ic3_logits"] is not None:
                c, t = top1_accuracy(out["ic3_logits"], batch["sic3"]); ic3_correct += c; ic3_total += t
            if out["ic4_logits"] is not None:
                c, t = top1_accuracy(out["ic4_logits"], batch["sic4"]); ic4_correct += c; ic4_total += t

    metrics = {
        "loss_mlm": loss_sums.get("mlm", 0)/max(1, counts.get("mlm", 1)),
        "loss_sop": loss_sums.get("sop", 0)/max(1, counts.get("sop", 1)),
        "loss_ic2": loss_sums.get("ic2", 0)/max(1, counts.get("ic2", 1)) if counts.get("ic2",0) else None,
        "loss_ic3": loss_sums.get("ic3", 0)/max(1, counts.get("ic3", 1)) if counts.get("ic3",0) else None,
        "loss_ic4": loss_sums.get("ic4", 0)/max(1, counts.get("ic4", 1)) if counts.get("ic4",0) else None,
        "loss_consistency": loss_sums.get("consistency", 0)/max(1, counts.get("consistency", 1)) if counts.get("consistency",0) else None,
        "acc_mlm": mlm_correct/max(1, mlm_total),
        "acc_sop": sop_correct/max(1, sop_total),
        "acc_ic2": ic2_correct/max(1, ic2_total) if ic2_total>0 else None,
        "acc_ic3": ic3_correct/max(1, ic3_total) if ic3_total>0 else None,
        "acc_ic4": ic4_correct/max(1, ic4_total) if ic4_total>0 else None,
    }
    print(
        f"[{desc}] "
        f"MLM loss {metrics['loss_mlm']:.4f}, acc {metrics['acc_mlm']:.4f} | "
        f"SOP loss {metrics['loss_sop']:.4f}, acc {metrics['acc_sop']:.4f} | "
        + (f"IC2 loss {metrics['loss_ic2']:.4f}, acc {metrics['acc_ic2']:.4f} | " if metrics['loss_ic2'] is not None else "")
        + (f"IC3 loss {metrics['loss_ic3']:.4f}, acc {metrics['acc_ic3']:.4f} | " if metrics['loss_ic3'] is not None else "")
        + (f"IC4 loss {metrics['loss_ic4']:.4f}, acc {metrics['acc_ic4']:.4f} | " if metrics['loss_ic4'] is not None else "")
        + (f"CONS {metrics['loss_consistency']:.4f}" if metrics['loss_consistency'] is not None else "")
    )
    return metrics