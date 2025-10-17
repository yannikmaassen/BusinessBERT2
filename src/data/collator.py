from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class Collator:
    # TODO: check for correct tokenizer type
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15  # Default to standard BERT value
    mask_probability: float = 0.8  # 80% of masked tokens are [MASK]
    random_probability: float = 0.1  # 10% of masked tokens are random tokens
    # The remaining 10% of masked tokens are kept unchanged

    def mask_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        labels = input_ids.clone()
        # Create probability matrix for deciding which tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device)

        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Decide which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for unmasked tokens (will be ignored in loss calculation)
        labels[~masked_indices] = -100

        # For masked tokens:
        # - 80% are replaced with [MASK]
        # - 10% are replaced with a random token
        # - 10% are left unchanged

        # Create a mask for the 80% to be replaced with [MASK]
        mask_indices = torch.bernoulli(torch.full(labels.shape, self.mask_probability, device=labels.device)).bool() & masked_indices
        input_ids[mask_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Create a mask for the 10% to be replaced with random tokens
        # This should be applied to the remaining 20% of masked tokens not yet converted to [MASK]
        remaining_indices = masked_indices & ~mask_indices
        random_indices = torch.bernoulli(torch.full(labels.shape, self.random_probability / (1 - self.mask_probability), device=labels.device)).bool() & remaining_indices

        # Generate random words for the 10% case
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        input_ids[random_indices] = random_words[random_indices]

        # The remaining 10% are kept unchanged - no action needed since we only modified the other 90%

        return input_ids, labels

    # TODO: IC
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert lists to tensors
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_token_type = [torch.tensor(f["token_type_ids"], dtype=torch.long) for f in features]
        batch_attention = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        sop = torch.tensor([f["sop_label"] for f in features], dtype=torch.long)
        sic2 = torch.tensor([f["sic2"] for f in features], dtype=torch.long)
        sic3 = torch.tensor([f["sic3"] for f in features], dtype=torch.long)
        sic4 = torch.tensor([f["sic4"] for f in features], dtype=torch.long)

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0)
        input_ids, mlm_labels = self.mask_tokens(input_ids)

        # Return batch
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