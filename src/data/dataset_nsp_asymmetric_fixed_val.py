import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetWithNSPAsymmetricFixedEval(Dataset):
    """
    Asymmetric NSP dataset (80/20 split) with fixed validation examples.
    Training: random sampling each epoch
    Validation: pre-built examples with fixed seed for reproducibility
    """

    def __init__(
            self,
            raw_examples: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            max_length: int,
            indexed_sic2_list: Dict[str, int],
            indexed_sic3_list: Dict[str, int],
            indexed_sic4_list: Dict[str, int],
            is_training: bool = True,
            seed: int = 42,
    ):
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexed_sic2_list = indexed_sic2_list
        self.indexed_sic3_list = indexed_sic3_list
        self.indexed_sic4_list = indexed_sic4_list
        self.is_training = is_training

        # 80/20 asymmetric split
        self.max_total_tokens = max_length - 3
        self.segment_a_budget = int(self.max_total_tokens * 0.8)
        self.segment_b_budget = self.max_total_tokens - self.segment_a_budget

        self.valid_examples = [
            ex for ex in raw_examples if ex.get("sentences")
        ]

        # Pre-build validation examples only
        if not is_training:
            self._prebuilt_examples = self._prebuild_validation_examples(seed)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.is_training:
            return self._get_random_example(idx)
        else:
            return self._get_prebuilt_example(idx)

    def _prebuild_validation_examples(self, seed: int) -> List[Dict[str, Any]]:
        """Pre-build validation examples with fixed random seed."""
        rng = random.Random(seed)
        prebuilt = []

        for example_a in tqdm(self.valid_examples, desc="Pre-building validation examples"):
            sentences = example_a.get("sentences", [])
            full_text_a = " ".join(sentences)
            tokenized_a = self.tokenizer(full_text_a, add_special_tokens=False)["input_ids"]

            is_next = rng.random() < 0.5

            if is_next:
                total_budget = self.segment_a_budget + self.segment_b_budget

                if len(tokenized_a) <= total_budget:
                    window = tokenized_a
                else:
                    max_start = len(tokenized_a) - total_budget
                    start = rng.randint(0, max_start)
                    window = tokenized_a[start:start + total_budget]

                a_ids = window[:self.segment_a_budget]
                b_ids = window[self.segment_a_budget:self.segment_a_budget + self.segment_b_budget]
                nsp_label = 0
                meta_example = example_a
            else:
                if len(tokenized_a) <= self.segment_a_budget:
                    a_ids = tokenized_a
                else:
                    max_start = len(tokenized_a) - self.segment_a_budget
                    start = rng.randint(0, max_start)
                    a_ids = tokenized_a[start:start + self.segment_a_budget]

                example_b = rng.choice(self.valid_examples)
                while example_b is example_a and len(self.valid_examples) > 1:
                    example_b = rng.choice(self.valid_examples)

                b_sentences = example_b.get("sentences", [])
                full_text_b = " ".join(b_sentences)
                tokenized_b = self.tokenizer(full_text_b, add_special_tokens=False)["input_ids"]

                if len(tokenized_b) <= self.segment_b_budget:
                    b_ids = tokenized_b
                else:
                    max_start = len(tokenized_b) - self.segment_b_budget
                    start = rng.randint(0, max_start)
                    b_ids = tokenized_b[start:start + self.segment_b_budget]

                nsp_label = 1
                meta_example = example_a

            prebuilt.append({
                'a_ids': a_ids,
                'b_ids': b_ids,
                'nsp_label': nsp_label,
                'meta_example': meta_example,
            })

        return prebuilt

    def _get_prebuilt_example(self, idx) -> Dict[str, Any]:
        """Return pre-built validation example."""
        prebuilt = self._prebuilt_examples[idx]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            prebuilt['a_ids'], prebuilt['b_ids']
        )
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
            prebuilt['a_ids'], prebuilt['b_ids']
        )
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        meta = prebuilt['meta_example']

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "nsp_labels": prebuilt['nsp_label'],
            "sic2": self._map_raw_sic_code_to_index(meta.get("sic2"), self.indexed_sic2_list),
            "sic3": self._map_raw_sic_code_to_index(meta.get("sic3"), self.indexed_sic3_list),
            "sic4": self._map_raw_sic_code_to_index(meta.get("sic4"), self.indexed_sic4_list),
        }

    def _get_random_example(self, idx) -> Dict[str, Any]:
        """Training: random sampling (original asymmetric logic)."""
        example_a = self.valid_examples[idx]
        sentences = example_a.get("sentences", [])
        full_text_a = " ".join(sentences)
        tokenized_a = self.tokenizer(full_text_a, add_special_tokens=False)["input_ids"]

        is_next = random.random() < 0.5

        if is_next:
            total_budget = self.segment_a_budget + self.segment_b_budget

            if len(tokenized_a) <= total_budget:
                window = tokenized_a
            else:
                max_start = len(tokenized_a) - total_budget
                start = random.randint(0, max_start)
                window = tokenized_a[start:start + total_budget]

            a_ids = window[:self.segment_a_budget]
            b_ids = window[self.segment_a_budget:self.segment_a_budget + self.segment_b_budget]
            nsp_label = 0
            meta_example = example_a
        else:
            if len(tokenized_a) <= self.segment_a_budget:
                a_ids = tokenized_a
            else:
                max_start = len(tokenized_a) - self.segment_a_budget
                start = random.randint(0, max_start)
                a_ids = tokenized_a[start:start + self.segment_a_budget]

            example_b = random.choice(self.valid_examples)
            while example_b is example_a and len(self.valid_examples) > 1:
                example_b = random.choice(self.valid_examples)

            b_sentences = example_b.get("sentences", [])
            full_text_b = " ".join(b_sentences)
            tokenized_b = self.tokenizer(full_text_b, add_special_tokens=False)["input_ids"]

            if len(tokenized_b) <= self.segment_b_budget:
                b_ids = tokenized_b
            else:
                max_start = len(tokenized_b) - self.segment_b_budget
                start = random.randint(0, max_start)
                b_ids = tokenized_b[start:start + self.segment_b_budget]

            nsp_label = 1
            meta_example = example_a

        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(a_ids, b_ids)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

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
