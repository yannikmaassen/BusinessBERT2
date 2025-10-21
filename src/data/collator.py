from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling


@dataclass
class BusinessBERTDataCollator:
    """
    Data collator that uses Hugging Face's DataCollatorForLanguageModeling for MLM
    while also handling industry classification labels.
    """
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        # Create the Hugging Face MLM collator
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract industry classification labels
        sic2 = torch.tensor([f.get("sic2", -100) for f in features], dtype=torch.long)
        sic3 = torch.tensor([f.get("sic3", -100) for f in features], dtype=torch.long)
        sic4 = torch.tensor([f.get("sic4", -100) for f in features], dtype=torch.long)
        sop_labels = torch.tensor([f.get("sop_label", 0) for f in features], dtype=torch.long)

        # Create batch with input features
        batch = {
            "input_ids": [torch.tensor(f["input_ids"], dtype=torch.long) for f in features],
            "token_type_ids": [torch.tensor(f.get("token_type_ids", [0] * len(f["input_ids"])), dtype=torch.long) for f in features],
            "attention_mask": [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features],
        }

        # Pad sequences
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=pad_token_id
        )
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            batch["token_type_ids"], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch["attention_mask"], batch_first=True, padding_value=0
        )

        # Apply MLM using Hugging Face's collator
        # Create dummy dataset entries with the required fields for the MLM collator
        mlm_inputs = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(input_ids, attention_mask)
        ]

        # Apply the MLM collator to get masked inputs and labels
        mlm_outputs = self.mlm_collator(mlm_inputs)

        # Return the combined batch with both MLM and industry classification
        return {
            "input_ids": mlm_outputs["input_ids"],
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": mlm_outputs["labels"],  # MLM labels
            "next_sentence_label": sop_labels,  # Renamed from sop_labels to match HF naming
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }