from typing import Tuple
import torch


@torch.no_grad()
def mlm_accuracy(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> Tuple[int, int]:
    preds = mlm_logits.argmax(dim=-1)
    mask = mlm_labels != -100
    correct = (preds[mask] == mlm_labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct, total