from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertPreTrainedModel, BertForPreTraining


def _kl_div(p_log, q, eps: float = 1e-8):
    """
    KL(q || exp(p_log)) with batchmean reduction.
    p_log: log probabilities (e.g., log softmax outputs)
    q: target probabilities (sum to 1 over last dim)
    """
    return F.kl_div(p_log, q.clamp(min=eps), reduction="batchmean")


class BusinessBERT2Pretrain(BertPreTrainedModel):
    """
    BERT encoder with:
      - MLM (token-level)
      - IC hierarchical (SIC2/3/4) + upward consistency (SIC4→SIC3 and SIC4→SIC2 via KL)
        * Multi-level cross-entropy at SIC2, SIC3, SIC4
        * Consistency encourages ancestor heads (SIC2/SIC3) to match leaf-implied marginals from SIC4
    """

    def __init__(
        self,
        config: BertConfig,
        n_sic2_classes: int,
        n_sic3_classes: int,
        n_sic4_classes: int,
        A32: torch.Tensor,  # [|SIC3| x |SIC2|] child->parent indicator (3->2)
        A43: torch.Tensor,  # [|SIC4| x |SIC3|] child->parent indicator (4->3)
        loss_weights: Dict[str, float],
    ):
        super().__init__(config)
        self.bert = BertForPreTraining(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.head_sic2 = nn.Linear(config.hidden_size, n_sic2_classes)
        self.head_sic3 = nn.Linear(config.hidden_size, n_sic3_classes)
        self.head_sic4 = nn.Linear(config.hidden_size, n_sic4_classes)

        # register upward mapping buffers
        # M43: [|SIC4| x |SIC3|] one-hot child->parent to sum leaf probs upward to SIC3
        # M42: [|SIC4| x |SIC2|] = A43 @ A32 to sum leaf probs upward to SIC2
        if A43.numel():
            M43 = A43.clone().to(torch.float32).contiguous()
        else:
            M43 = torch.empty(0)

        if A43.numel() and A32.numel():
            M42 = torch.matmul(A43.to(torch.float32), A32.to(torch.float32)).contiguous()
        else:
            M42 = torch.empty(0)

        self.register_buffer("M43", M43)  # [|SIC4| x |SIC3|]
        self.register_buffer("M42", M42)  # [|SIC4| x |SIC2|]

        # Register child-to-parent matrices as buffers (non-trainable)
        self.register_buffer("child_to_parent_matrix_sic4_to_sic3", M43)
        self.register_buffer("child_to_parent_matrix_sic4_to_sic2", M42)

        self.loss_weights = dict(loss_weights)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            mlm_labels: Optional[torch.Tensor] = None,
            sic2: Optional[torch.Tensor] = None,
            sic3: Optional[torch.Tensor] = None,
            sic4: Optional[torch.Tensor] = None,
    ):
        base_model = self.bert.bert
        transformer_outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = transformer_outputs.last_hidden_state
        pooled_output = self.dropout(transformer_outputs.pooler_output)

        mlm_logits = self.bert.cls.predictions(sequence_output)

        sic2_logits = self.head_sic2(pooled_output) if self.head_sic2 is not None else None
        sic3_logits = self.head_sic3(pooled_output) if self.head_sic3 is not None else None
        sic4_logits = self.head_sic4(pooled_output) if self.head_sic4 is not None else None

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        total_loss = 0.0

        # ----- MLM -----
        if mlm_labels is not None:
            mlm_loss = self.ce(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            losses["mlm"] = mlm_loss
            total_loss = total_loss + self.loss_weights.get("mlm", 1.0) * mlm_loss

            # MLM accuracy (only on masked tokens, ignore -100)
            mask = mlm_labels != -100
            if mask.any():
                predictions = mlm_logits.argmax(dim=-1)
                correct = (predictions == mlm_labels) & mask
                metrics["mlm_accuracy"] = correct.sum().float() / mask.sum().float()

        # ----- IC cross-entropy at each level -----
        if sic2_logits is not None and sic2 is not None:
            ic2_loss = self.ce(sic2_logits, sic2)
            losses["ic2"] = ic2_loss
            total_loss += self.loss_weights.get("ic2", 0.0) * ic2_loss

            # SIC2 accuracy
            mask = sic2 != -100
            if mask.any():
                predictions = sic2_logits.argmax(dim=-1)
                correct = (predictions == sic2) & mask
                metrics["ic2_accuracy"] = correct.sum().float() / mask.sum().float()

        if sic3_logits is not None and sic3 is not None:
            ic3_loss = self.ce(sic3_logits, sic3)
            losses["ic3"] = ic3_loss
            total_loss += self.loss_weights.get("ic3", 0.0) * ic3_loss

            # SIC3 accuracy
            mask = sic3 != -100
            if mask.any():
                predictions = sic3_logits.argmax(dim=-1)
                correct = (predictions == sic3) & mask
                metrics["ic3_accuracy"] = correct.sum().float() / mask.sum().float()

        if sic4_logits is not None and sic4 is not None:
            ic4_loss = self.ce(sic4_logits, sic4)
            losses["ic4"] = ic4_loss
            total_loss += self.loss_weights.get("ic4", 0.0) * ic4_loss

            # SIC4 accuracy
            mask = sic4 != -100
            if mask.any():
                predictions = sic4_logits.argmax(dim=-1)
                correct = (predictions == sic4) & mask
                metrics["ic4_accuracy"] = correct.sum().float() / mask.sum().float()

        # ----- Upward consistency from SIC4 → SIC3 and SIC4 → SIC2 -----
        have_m43 = (sic4_logits is not None) and (sic3_logits is not None) and (
                    self.child_to_parent_matrix_sic4_to_sic3.numel() > 0)
        have_m42 = (sic4_logits is not None) and (sic2_logits is not None) and (
                    self.child_to_parent_matrix_sic4_to_sic2.numel() > 0)

        if have_m43 or have_m42:
            parts = []
            eps = 1e-8

            p4 = F.softmax(sic4_logits, dim=-1)  # [B, n4]

            if have_m43:
                # implied SIC3 distribution by summing SIC4 children
                implied_p3 = torch.matmul(p4, self.child_to_parent_matrix_sic4_to_sic3)  # [B, n3]
                log_p3 = F.log_softmax(sic3_logits, dim=-1)
                # KL(implied || predicted) to encourage predicted to match implied
                parts.append(_kl_div(log_p3, implied_p3))

            if have_m42:
                # implied SIC2 distribution by summing SIC4 children
                implied_p2 = torch.matmul(p4, self.child_to_parent_matrix_sic4_to_sic2)  # [B, n2]
                log_p2 = F.log_softmax(sic2_logits, dim=-1)
                # KL(implied || predicted) to encourage predicted to match implied
                parts.append(_kl_div(log_p2, implied_p2))

            if parts:
                consistency_loss = torch.stack(parts).mean()
                losses["consistency"] = consistency_loss
                total_loss += self.loss_weights.get("consistency", 0.0) * consistency_loss

        return {
            "loss": total_loss,
            "losses": losses,
            "metrics": metrics,
            "mlm_logits": mlm_logits,
            "ic2_logits": sic2_logits,
            "ic3_logits": sic3_logits,
            "ic4_logits": sic4_logits,
        }
