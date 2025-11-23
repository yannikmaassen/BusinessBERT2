import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetWithNSPAsymmetric(Dataset):
    """
    Asymmetric segment allocation for better CLS representation.
    Allocates segment_a_ratio (e.g., 0.6-0.8) of tokens to segment A,
    remainder to segment B. Always fills to max_length for optimal MLM.
    """

    def __init__(
            self,
            raw_examples: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            max_length: int,
            indexed_sic2_list: Dict[str, int],
            indexed_sic3_list: Dict[str, int],
            indexed_sic4_list: Dict[str, int],
            segment_a_ratio: float = 0.8,
    ):
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexed_sic2_list = indexed_sic2_list
        self.indexed_sic3_list = indexed_sic3_list
        self.indexed_sic4_list = indexed_sic4_list

        self.segment_a_ratio = segment_a_ratio

        # Budget excluding special tokens ([CLS], [SEP], [SEP])
        total_budget = self.max_length - 3
        self.segment_a_budget = int(total_budget * segment_a_ratio)
        self.segment_b_budget = total_budget - self.segment_a_budget

        self.valid_examples = [
            ex for ex in tqdm(raw_examples, desc="Filtering valid examples")
            if ex.get("sentences")
        ]

    def _get_full_text_from_example(self, ex: Dict[str, Any]) -> str:
        sentences = ex.get("sentences") or []
        return " ".join(sentences)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example_a = self.valid_examples[idx]
        full_text_a = self._get_full_text_from_example(example_a)

        # Tokenize full document on-the-fly (no special tokens)
        tokenized_a = self.tokenizer(full_text_a, add_special_tokens=False)["input_ids"]

        is_next = random.random() < 0.5

        if is_next:
            # Positive: sample window, then split asymmetrically
            total_budget = self.segment_a_budget + self.segment_b_budget

            if len(tokenized_a) <= total_budget:
                window = tokenized_a
            else:
                max_start = len(tokenized_a) - total_budget
                start = random.randint(0, max_start)
                window = tokenized_a[start:start + total_budget]

            # Asymmetric split
            a_ids = window[:self.segment_a_budget]
            b_ids = window[self.segment_a_budget:]

            nsp_label = 0
            meta_example = example_a
        else:
            # Negative: A from current doc, B from different doc
            if len(tokenized_a) <= self.segment_a_budget:
                a_ids = tokenized_a
            else:
                max_start = len(tokenized_a) - self.segment_a_budget
                start = random.randint(0, max_start)
                a_ids = tokenized_a[start:start + self.segment_a_budget]

            # Get segment B from different document
            example_b = random.choice(self.valid_examples)
            while example_b is example_a and len(self.valid_examples) > 1:
                example_b = random.choice(self.valid_examples)

            full_text_b = self._get_full_text_from_example(example_b)
            tokenized_b = self.tokenizer(full_text_b, add_special_tokens=False)["input_ids"]

            if len(tokenized_b) <= self.segment_b_budget:
                b_ids = tokenized_b
            else:
                max_start = len(tokenized_b) - self.segment_b_budget
                start = random.randint(0, max_start)
                b_ids = tokenized_b[start:start + self.segment_b_budget]

            nsp_label = 1
            meta_example = example_a

        # Build final inputs with special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(a_ids, b_ids)
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len

        sic2 = self._map_raw_sic_code_to_index(meta_example.get("sic2"), self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(meta_example.get("sic3"), self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(meta_example.get("sic4"), self.indexed_sic4_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "nsp_labels": nsp_label,
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
