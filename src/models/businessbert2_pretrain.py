from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel
from src.models.heads import BertPretrainHeads


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
      - SOP (binary)
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
        self.bert = BertModel(config)
        self.heads = BertPretrainHeads(config, self.bert.get_input_embeddings())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.head_sic2 = nn.Linear(config.hidden_size, n_sic2_classes)
        self.head_sic3 = nn.Linear(config.hidden_size, n_sic3_classes)
        self.head_sic4 = nn.Linear(config.hidden_size, n_sic4_classes)

        # ----- NEW: register upward mapping buffers -----
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
        self.register_buffer(
            "child_to_parent_matrix_sic4_to_sic3", M43
        )
        self.register_buffer(
            "child_to_parent_matrix_sic4_to_sic2", M42
        )

        self.loss_weights = dict(loss_weights)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_weights()


    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        mlm_labels: Optional[torch.Tensor] = None,
        sop_labels: Optional[torch.Tensor] = None,
        sic2: Optional[torch.Tensor] = None,
        sic3: Optional[torch.Tensor] = None,
        sic4: Optional[torch.Tensor] = None,
    ):
        transformer_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = transformer_outputs.last_hidden_state
        pooled_output = self.dropout(transformer_outputs.pooler_output)

        mlm_logits, sop_logits = self.heads(sequence_output, pooled_output)

        sic2_logits = self.head_sic2(pooled_output) if self.head_sic2 is not None else None  # [batch, n2]
        sic3_logits = self.head_sic3(pooled_output) if self.head_sic3 is not None else None  # [batch, n3]
        sic4_logits = self.head_sic4(pooled_output) if self.head_sic4 is not None else None # [batch, n4]

        losses: Dict[str, torch.Tensor] = {}
        total_loss = 0.0

        # ----- MLM -----
        if mlm_labels is not None:
            mlm_loss = self.ce(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            losses["mlm"] = mlm_loss
            total_loss = total_loss + self.loss_weights.get("mlm", 1.0) * mlm_loss

        # ----- SOP -----
        if sop_labels is not None:
            sop_loss = self.ce(sop_logits, sop_labels)
            losses["sop"] = sop_loss
            total_loss = total_loss + self.loss_weights.get("sop", 1.0) * sop_loss

        # ----- IC cross-entropy at each level -----
        if sic2_logits is not None and sic2 is not None:
            ic2_loss = self.ce(sic2_logits, sic2)
            losses["ic2"] = ic2_loss
            total_loss += self.loss_weights.get("ic2", 0.0) * ic2_loss

        if sic3_logits is not None and sic3 is not None:
            ic3_loss = self.ce(sic3_logits, sic3)
            losses["ic3"] = ic3_loss
            total_loss += self.loss_weights.get("ic3", 0.0) * ic3_loss

        if sic4_logits is not None and sic4 is not None:
            ic4_loss = self.ce(sic4_logits, sic4)
            losses["ic4"] = ic4_loss
            total_loss += self.loss_weights.get("ic4", 0.0) * ic4_loss

        # ----- Upward consistency from SIC4 → SIC3 and SIC4 → SIC2 -----
        have_m43 = (sic4_logits is not None) and (sic3_logits is not None) and (self.child_to_parent_matrix_sic4_to_sic3.numel() > 0)
        have_m42 = (sic4_logits is not None) and (sic2_logits is not None) and (self.child_to_parent_matrix_sic4_to_sic2.numel() > 0)

        if have_m43 or have_m42:
            parts = []
            eps = 1e-8

            p4 = F.softmax(sic4_logits, dim=-1)  # [B, n4]

            if have_m43:
                # implied SIC3 distribution by summing SIC4 children
                implied_p3 = torch.matmul(p4, self.child_to_parent_matrix_sic4_to_sic3)  # [B, n3], sums to 1
                p3 = F.softmax(sic3_logits, dim=-1)
                parts.append(_kl_div(p3.clamp_min(eps).log(), implied_p3))

            if have_m42:
                # implied SIC2 distribution by summing SIC4 children (via 4->3->2)
                implied_p2 = torch.matmul(p4, self.child_to_parent_matrix_sic4_to_sic2)  # [B, n2], sums to 1
                p2 = F.softmax(sic2_logits, dim=-1)
                parts.append(_kl_div(p2.clamp_min(eps).log(), implied_p2))

            if parts:
                consistency_loss = torch.stack(parts).mean()
                losses["consistency"] = consistency_loss
                total_loss += self.loss_weights.get("consistency", 0.0) * consistency_loss

        return {
            "loss": total_loss,
            "losses": losses,
            "mlm_logits": mlm_logits,
            "sop_logits": sop_logits,
            "ic2_logits": sic2_logits,
            "ic3_logits": sic3_logits,
            "ic4_logits": sic4_logits,
        }