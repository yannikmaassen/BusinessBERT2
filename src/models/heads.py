import torch.nn as nn
from transformers import BertConfig


class BertPretrainHeads(nn.Module):
    """MLM + SOP heads with tied embeddings."""
    def __init__(self, config: BertConfig, embedding_weights: nn.Embedding):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(nn.init.zeros_(nn.Parameter(nn.zeros(config.vocab_size))).data)
        self.mlm_decoder.weight = embedding_weights.weight
        self.sop_classifier = nn.Linear(config.hidden_size, 2)


    def forward(self, sequence_output, pooled_output):
        mlm_hidden = self.transform(sequence_output)
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias
        sop_logits = self.sop_classifier(pooled_output)

        return mlm_logits, sop_logits