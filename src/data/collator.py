from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15  # Default to standard BERT value

    def __post_init__(self):
        # Use Hugging Face's data collator for MLM
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
            return_tensors="pt"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input features
        batch = {
            "input_ids": [torch.tensor(f["input_ids"], dtype=torch.long) for f in features],
            "token_type_ids": [torch.tensor(f["token_type_ids"], dtype=torch.long) for f in features],
            "attention_mask": [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features],
        }

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            batch["token_type_ids"], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch["attention_mask"], batch_first=True, padding_value=0
        )

        # Apply MLM using Hugging Face's collator
        mlm_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        mlm_batch = self.mlm_collator([{k: v for k, v in zip(mlm_inputs.keys(), values)}
                                       for values in zip(*(mlm_inputs[k] for k in mlm_inputs.keys()))])

        # Extract masked input_ids and labels
        input_ids = mlm_batch["input_ids"]
        mlm_labels = mlm_batch["labels"]  # -100 for unmasked tokens

        # Extract other classification labels
        sop = torch.tensor([f["sop_label"] for f in features], dtype=torch.long)
        sic2 = torch.tensor([f["sic2"] for f in features], dtype=torch.long)
        sic3 = torch.tensor([f["sic3"] for f in features], dtype=torch.long)
        sic4 = torch.tensor([f["sic4"] for f in features], dtype=torch.long)


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