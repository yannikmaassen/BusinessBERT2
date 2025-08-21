from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel
from .heads import BertPretrainHeads


def _kl_div(p_log, q, eps: float = 1e-8):
    return F.kl_div(p_log, q.clamp(min=eps), reduction="batchmean")


def _js_div(p, q, eps: float = 1e-8):
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p.log(), m, eps) + 0.5 * _kl_div(q.log(), m, eps)


class BusinessBERT2Pretrain(BertPreTrainedModel):
    """
    BERT encoder with:
      - MLM (token-level)
      - SOP (binary)
      - IC hierarchical (SIC2/3/4) + JS-based downward consistency (SIC2→SIC3, SIC3→SIC4)
    """

    def __init__(
        self,
        config: BertConfig,
        n_sic2: int,
        n_sic3: int,
        n_sic4: int,
        A32: torch.Tensor,
        A43: torch.Tensor,
        loss_weights: Dict[str, float],
    ):
        super().__init__(config)
        self.bert = BertModel(config)
        self.heads = BertPretrainHeads(config, self.bert.get_input_embeddings())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ic2 = nn.Linear(config.hidden_size, n_sic2) if n_sic2 > 0 else None
        self.ic3 = nn.Linear(config.hidden_size, n_sic3) if n_sic3 > 0 else None
        self.ic4 = nn.Linear(config.hidden_size, n_sic4) if n_sic4 > 0 else None

        B23 = A32.T.clone()
        B34 = A43.T.clone()
        if B23.numel():
            B23 = (B23 / (B23.sum(dim=1, keepdim=True).clamp(min=1e-12))).contiguous()
        if B34.numel():
            B34 = (B34 / (B34.sum(dim=1, keepdim=True).clamp(min=1e-12))).contiguous()
        self.register_buffer("B23", B23)  # [|SIC2| x |SIC3|]
        self.register_buffer("B34", B34)  # [|SIC3| x |SIC4|]

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
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = out.last_hidden_state
        pooled_output = self.dropout(out.pooler_output)

        mlm_logits, sop_logits = self.heads(sequence_output, pooled_output)

        ic2_logits = self.ic2(pooled_output) if self.ic2 is not None else None
        ic3_logits = self.ic3(pooled_output) if self.ic3 is not None else None
        ic4_logits = self.ic4(pooled_output) if self.ic4 is not None else None

        losses: Dict[str, torch.Tensor] = {}
        total_loss = 0.0

        if mlm_labels is not None:
            mlm_loss = self.ce(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            losses["mlm"] = mlm_loss
            total_loss = total_loss + self.loss_weights.get("mlm", 1.0) * mlm_loss

        if sop_labels is not None:
            sop_loss = self.ce(sop_logits, sop_labels)
            losses["sop"] = sop_loss
            total_loss = total_loss + self.loss_weights.get("sop", 1.0) * sop_loss

        if ic2_logits is not None and sic2 is not None:
            ic2_loss = self.ce(ic2_logits, sic2)
            losses["ic2"] = ic2_loss
            total_loss += self.loss_weights.get("ic2", 0.0) * ic2_loss

        if ic3_logits is not None and sic3 is not None:
            ic3_loss = self.ce(ic3_logits, sic3)
            losses["ic3"] = ic3_loss
            total_loss += self.loss_weights.get("ic3", 0.0) * ic3_loss

        if ic4_logits is not None and sic4 is not None:
            ic4_loss = self.ce(ic4_logits, sic4)
            losses["ic4"] = ic4_loss
            total_loss += self.loss_weights.get("ic4", 0.0) * ic4_loss

        # Downward consistency: SIC2→SIC3 and SIC3→SIC4
        have23 = (ic2_logits is not None) and (ic3_logits is not None) and (self.B23.numel() > 0)
        have34 = (ic3_logits is not None) and (ic4_logits is not None) and (self.B34.numel() > 0)
        if have23 or have34:
            cons_parts = []
            if have23:
                p2 = F.softmax(ic2_logits, dim=-1)
                implied_p3 = torch.matmul(p2, self.B23)
                p3 = F.softmax(ic3_logits, dim=-1)
                cons_parts.append(_js_div(implied_p3, p3))
            if have34:
                p3 = F.softmax(ic3_logits, dim=-1)
                implied_p4 = torch.matmul(p3, self.B34)
                p4 = F.softmax(ic4_logits, dim=-1)
                cons_parts.append(_js_div(implied_p4, p4))
            if cons_parts:
                consistency_loss = torch.stack(cons_parts).mean()
                losses["consistency"] = consistency_loss
                total_loss += self.loss_weights.get("consistency", 0.0) * consistency_loss

        return {
            "loss": total_loss,
            "losses": losses,
            "mlm_logits": mlm_logits,
            "sop_logits": sop_logits,
            "ic2_logits": ic2_logits,
            "ic3_logits": ic3_logits,
            "ic4_logits": ic4_logits,
        }