from typing import Tuple
import torch


@torch.no_grad()
def mlm_accuracy(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> Tuple[int, int]:
    predictions = mlm_logits.argmax(dim=-1)
    mask = mlm_labels != -100
    correct = (predictions[mask] == mlm_labels[mask]).sum().item()
    total = mask.sum().item()

    return correct, total


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0, 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()

    return correct, total