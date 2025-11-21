from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
)

@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        self._mlm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pull out custom labels first (so the MLM collator ignores them)
        extra_keys = ["nsp_labels", "sic2", "sic3", "sic4"]

        extras = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                  for k in extra_keys if k in features[0]}

        base_features = [{k: v for k, v in f.items() if k not in extra_keys}
                         for f in features]

        # HF handles padding + MLM masking (returns 'input_ids', 'attention_mask', 'labels')
        batch = self._mlm(base_features)

        # Rename to match expected output
        batch["mlm_labels"] = batch.pop("labels")

        for k in extra_keys:
            if k in extras:
                batch[k] = extras[k]

        return batch
