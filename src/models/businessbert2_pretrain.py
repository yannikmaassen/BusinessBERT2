from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForPreTraining


def _kl_div(p_log, q, eps: float = 1e-8):
    """
    KL(q || exp(p_log)) with batchmean reduction.
    p_log: log probabilities (e.g., log softmax outputs)
    q: target probabilities (sum to 1 over last dim)
    """
    return F.kl_div(p_log, q.clamp(min=eps), reduction="batchmean")


class BusinessBERT2Pretrain(BertForPreTraining):
    """
    BERT pretraining model with MLM and NSP from BertForPreTraining, plus:
    - IC hierarchical (SIC2/3/4) + upward consistency (SIC4→SIC3 and SIC4→SIC2 via KL)
      * Multi-level cross-entropy at SIC2, SIC3, SIC4
      * Consistency encourages ancestor heads (SIC2/SIC3) to match leaf-implied marginals from SIC4
    """

    def __init__(
        self,
        config,
        n_sic2_classes: int = 0,
        n_sic3_classes: int = 0,
        n_sic4_classes: int = 0,
        A32: Optional[torch.Tensor] = None,  # [|SIC3| x |SIC2|] child->parent indicator (3->2)
        A43: Optional[torch.Tensor] = None,  # [|SIC4| x |SIC3|] child->parent indicator (4->3)
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(config)
        # Use BertForPreTraining's MLM and NSP heads

        # Add industry classification heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_sic2 = nn.Linear(config.hidden_size, n_sic2_classes) if n_sic2_classes > 0 else None
        self.head_sic3 = nn.Linear(config.hidden_size, n_sic3_classes) if n_sic3_classes > 0 else None
        self.head_sic4 = nn.Linear(config.hidden_size, n_sic4_classes) if n_sic4_classes > 0 else None

        # Register upward mapping buffers if needed
        self.child_to_parent_matrix_sic4_to_sic3 = None
        self.child_to_parent_matrix_sic4_to_sic2 = None

        if A43 is not None and A43.numel() > 0:
            M43 = A43.clone().to(torch.float32).contiguous()
            self.register_buffer("child_to_parent_matrix_sic4_to_sic3", M43)

            if A32 is not None and A32.numel() > 0:
                M42 = torch.matmul(A43.to(torch.float32), A32.to(torch.float32)).contiguous()
                self.register_buffer("child_to_parent_matrix_sic4_to_sic2", M42)

        self.loss_weights = loss_weights or {
            "mlm": 1.0,
            "nsp": 1.0,
            "ic2": 0.0,
            "ic3": 0.0,
            "ic4": 0.0,
            "consistency": 0.0
        }
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,  # MLM labels
        next_sentence_label=None,  # NSP label
        sic2=None,  # SIC2 label
        sic3=None,  # SIC3 label
        sic4=None,  # SIC4 label
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # First, get the standard BERT pretraining outputs (MLM and NSP)
        bert_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # The base loss from BERT pretraining (MLM + NSP)
        bert_loss = bert_outputs.loss if bert_outputs.loss is not None else 0.0

        # Track individual losses
        losses = {}
        if labels is not None:
            mlm_loss = self.ce(
                bert_outputs.prediction_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            losses["mlm"] = mlm_loss

        if next_sentence_label is not None:
            nsp_loss = self.ce(
                bert_outputs.seq_relationship_logits,
                next_sentence_label
            )
            losses["sop"] = nsp_loss  # Using 'sop' for consistency with original code

        # Apply loss weights to MLM and NSP
        total_loss = (
            self.loss_weights.get("mlm", 1.0) * losses.get("mlm", 0.0) +
            self.loss_weights.get("sop", 1.0) * losses.get("sop", 0.0)
        )

        # If no industry classification tasks are used, just return BERT pretraining outputs
        if sic2 is None and sic3 is None and sic4 is None:
            if not return_dict:
                return (total_loss,) + bert_outputs[1:]

            result = bert_outputs.to_tuple()
            result = (total_loss,) + result[1:]
            return result

        # Add industry classification tasks
        # Get pooled output for classification
        pooled_output = self.dropout(self.bert.pooler(
            self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            ).last_hidden_state
        ))

        # Industry classification logits
        sic2_logits = self.head_sic2(pooled_output) if self.head_sic2 is not None else None
        sic3_logits = self.head_sic3(pooled_output) if self.head_sic3 is not None else None
        sic4_logits = self.head_sic4(pooled_output) if self.head_sic4 is not None else None

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
        have_m43 = (sic4_logits is not None) and (sic3_logits is not None) and hasattr(self, "child_to_parent_matrix_sic4_to_sic3") and self.child_to_parent_matrix_sic4_to_sic3 is not None
        have_m42 = (sic4_logits is not None) and (sic2_logits is not None) and hasattr(self, "child_to_parent_matrix_sic4_to_sic2") and self.child_to_parent_matrix_sic4_to_sic2 is not None

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

        # Return everything in the format expected by the Trainer
        return {
            "loss": total_loss,
            "losses": losses,
            "logits": bert_outputs.prediction_logits,
            "seq_relationship_logits": bert_outputs.seq_relationship_logits,
            "ic2_logits": sic2_logits,
            "ic3_logits": sic3_logits,
            "ic4_logits": sic4_logits,
        }