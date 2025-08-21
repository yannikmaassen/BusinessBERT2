from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float
    rand_probability: float
    keep_probability: float

    def mask_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 1 - self.rand_probability - self.keep_probability, device=labels.device)
            ).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% -> random
        indices_random = (
            torch.bernoulli(
                torch.full(labels.shape, self.rand_probability / (self.rand_probability + self.keep_probability + 1e-8), device=labels.device)
            ).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_token_type = [torch.tensor(f["token_type_ids"], dtype=torch.long) for f in features]
        batch_attention = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        sop = torch.tensor([f["sop_label"] for f in features], dtype=torch.long)
        sic2 = torch.tensor([f["sic2"] for f in features], dtype=torch.long)
        sic3 = torch.tensor([f["sic3"] for f in features], dtype=torch.long)
        sic4 = torch.tensor([f["sic4"] for f in features], dtype=torch.long)

        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0)
        input_ids, mlm_labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "sop_labels": sop,
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }