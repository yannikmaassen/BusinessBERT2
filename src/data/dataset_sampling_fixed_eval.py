import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetOnTheFlyNewFixedEval(Dataset):
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
        self.seed = seed

        self.valid_examples: List[Dict[str, Any]] = [
            ex for ex in tqdm(raw_examples, desc="Filtering valid examples")
            if ex.get("sentences")
        ]

        if not self.is_training:
            self._prebuilt_examples = self._prebuild_validation_examples(self.seed)


    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.is_training:
            return self._get_random_example(idx)
        else:
            return self._get_prebuilt_example(idx)


    def _prebuild_validation_examples(self, seed: int) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        prebuilt: List[Dict[str, Any]] = []
        body_max_len = self.max_length - 2  # [CLS] + body + [SEP]

        for example in tqdm(self.valid_examples, desc="Pre-building validation examples"):
            sentences = example.get("sentences", [])
            full_text = " ".join(sentences)

            encoding = self.tokenizer(
                full_text,
                truncation=False,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            input_ids: List[int] = encoding["input_ids"]
            attention_mask: List[int] = encoding["attention_mask"]
            seq_len = len(input_ids)

            if seq_len > body_max_len:
                max_start = seq_len - body_max_len
                start_idx = rng.randint(0, max_start)
                body_ids = input_ids[start_idx:start_idx + body_max_len]
                body_mask = attention_mask[start_idx:start_idx + body_max_len]
            else:
                body_ids = input_ids
                body_mask = attention_mask

            prebuilt.append({
                "body_ids": body_ids,
                "body_mask": body_mask,
                "meta_example": example,
            })

        return prebuilt

    def _get_prebuilt_example(self, idx: int) -> Dict[str, Any]:
        item = self._prebuilt_examples[idx]
        body_ids = item["body_ids"]
        body_mask = item["body_mask"]
        meta = item["meta_example"]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(body_ids)
        attention_mask = [1] + body_mask + [1]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sic2": self._map_raw_sic_code_to_index(meta.get("sic2"), self.indexed_sic2_list),
            "sic3": self._map_raw_sic_code_to_index(meta.get("sic3"), self.indexed_sic3_list),
            "sic4": self._map_raw_sic_code_to_index(meta.get("sic4"), self.indexed_sic4_list),
        }

    def _get_random_example(self, idx: int) -> Dict[str, Any]:
        example = self.valid_examples[idx]
        sentences = example.get("sentences", [])
        full_text = " ".join(sentences)

        encoding = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        input_ids: List[int] = encoding["input_ids"]
        attention_mask: List[int] = encoding["attention_mask"]
        seq_len = len(input_ids)

        print(f"Length full encoding input ids: {len(input_ids)}")

        body_max_len = self.max_length - 2

        if seq_len > body_max_len:
            max_start = seq_len - body_max_len
            start_idx = random.randint(0, max_start)
            body_ids = input_ids[start_idx:start_idx + body_max_len]
            body_mask = attention_mask[start_idx:start_idx + body_max_len]
        else:
            body_ids = input_ids
            body_mask = attention_mask

        input_ids = self.tokenizer.build_inputs_with_special_tokens(body_ids)
        attention_mask = [1] + body_mask + [1]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            print(f"Padding length: {pad_len}")
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sic2": self._map_raw_sic_code_to_index(example.get("sic2"), self.indexed_sic2_list),
            "sic3": self._map_raw_sic_code_to_index(example.get("sic3"), self.indexed_sic3_list),
            "sic4": self._map_raw_sic_code_to_index(example.get("sic4"), self.indexed_sic4_list),
        }


    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100)

    def __len__(self) -> int:
        return len(self.valid_examples)
