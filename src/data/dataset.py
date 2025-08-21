from typing import Dict, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset):
    def __init__(self, examples, tokenizer: PreTrainedTokenizerBase, max_length: int,
                 idx2: Dict[str, int], idx3: Dict[str, int], idx4: Dict[str, int]):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]
        encoding = self.tokenizer(
            example.text_a if example.text_a else "",
            example.text_b if example.text_b else None,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        sic2 = self.idx2.get(example.sic2, -100)
        sic3 = self.idx3.get(example.sic3, -100)
        sic4 = self.idx4.get(example.sic4, -100)

        return {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
            "attention_mask": encoding["attention_mask"],
            "sop_label": example.sop_label,
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }