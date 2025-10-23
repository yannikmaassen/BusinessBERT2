import torch
from transformers import BertConfig
import math


class BertPretrainHeads(torch.nn.Module):
    def __init__(self, config: BertConfig, embedding_weights: torch.nn.Embedding):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.mlm_decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        init_value = 1.0 / math.sqrt(config.vocab_size)
        self.mlm_bias = torch.nn.Parameter(torch.ones(config.vocab_size) * init_value)
        self.mlm_decoder.weight = embedding_weights.weight
        self.nsp_classifier = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        mlm_hidden = self.transform(sequence_output)
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias
        nsp_logits = self.nsp_classifier(pooled_output)

        return mlm_logits, nsp_logits