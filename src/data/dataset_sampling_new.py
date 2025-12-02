import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

class PretrainDatasetOnTheFlyNew(Dataset):
    def __init__(
        self,
        raw_examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        indexed_sic2_list: Dict[str, int],
        indexed_sic3_list: Dict[str, int],
        indexed_sic4_list: Dict[str, int],
    ):
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexed_sic2_list = indexed_sic2_list
        self.indexed_sic3_list = indexed_sic3_list
        self.indexed_sic4_list = indexed_sic4_list

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.valid_examples = []
        for example in tqdm(raw_examples, desc="Filtering valid examples"):
            sentences = example.get("sentences", [])
            if sentences:
                self.valid_examples.append(example)


    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.valid_examples[idx]
        sentences = example.get("sentences", [])
        full_text = " ".join(sentences)

        encoding = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_special_tokens_mask=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        seq_len = len(input_ids)

        # reserve 2 tokens for [CLS] and [SEP]
        body_max_len = self.max_length - 2

        if seq_len > body_max_len:
            max_start = seq_len - body_max_len
            start_idx = random.randint(0, max_start)

            body_ids = input_ids[start_idx:start_idx + body_max_len]
            body_mask = attention_mask[start_idx:start_idx + body_max_len]
        else:
            body_ids = input_ids
            body_mask = attention_mask

        final_input_ids = (
            [self.cls_token_id] +
            body_ids +
            [self.sep_token_id]
        )

        final_attention_mask = (
            [1] + body_mask + [1]
        )

        sic2 = self._map_raw_sic_code_to_index(example.get("sic2"), self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(example.get("sic3"), self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(example.get("sic4"), self.indexed_sic4_list)

        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }


    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100)


    def __len__(self) -> int:
        return len(self.valid_examples)
