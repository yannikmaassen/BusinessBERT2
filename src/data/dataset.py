from typing import Dict, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset):
    def __init__(self, examples, tokenizer: PreTrainedTokenizerBase, max_len: int,
                 idx2: Dict[str, int], idx3: Dict[str, int], idx4: Dict[str, int]):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tok(
            ex.text_a if ex.text_a else "",
            ex.text_b if ex.text_b else None,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        sic2 = self.idx2.get(ex.sic2, -100)
        sic3 = self.idx3.get(ex.sic3, -100)
        sic4 = self.idx4.get(ex.sic4, -100)
        return {
            "input_ids": enc["input_ids"],
            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
            "attention_mask": enc["attention_mask"],
            "sop_label": ex.sop_label,
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }