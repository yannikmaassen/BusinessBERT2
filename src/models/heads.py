import torch
from transformers import BertConfig


class BertPretrainHeads(torch.nn.Module):
    """MLM + SOP heads with tied embeddings."""
    def __init__(self, config: BertConfig, embedding_weights: torch.nn.Embedding):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.mlm_decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_decoder.weight = embedding_weights.weight
        self.sop_classifier = torch.nn.Linear(config.hidden_size, 2)


    def forward(self, sequence_output, pooled_output):
        mlm_hidden = self.transform(sequence_output)
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias
        sop_logits = self.sop_classifier(pooled_output)

        return mlm_logits, sop_logits