import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetRandomSampling(Dataset):
    """
    Pretraining dataset with RANDOM SAMPLING strategy.

    For documents longer than max_length:
    - Samples ONE random window of max_length tokens per document
    - Resamples a different window each epoch for data augmentation
    """

    def __init__(
        self,
        raw_examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        indexed_sic2_list: Dict[str, int],
        indexed_sic3_list: Dict[str, int],
        indexed_sic4_list: Dict[str, int],
        preprocess_device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexed_sic2_list = indexed_sic2_list
        self.indexed_sic3_list = indexed_sic3_list
        self.indexed_sic4_list = indexed_sic4_list

        # Force preprocessing on CPU
        if hasattr(self.tokenizer, 'to'):
            self.tokenizer = self.tokenizer.to(preprocess_device)

        print("Preprocessing examples with random sampling...")
        self.examples = self._preprocess_examples(raw_examples)
        print(f"Created {len(self.examples)} training examples (1:1 ratio with raw data)")


    def _preprocess_examples(self, raw_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []

        skipped_empty = 0
        examples_within_max = 0
        examples_sampled = 0

        for idx, example in enumerate(tqdm(raw_examples, desc="Preprocessing examples")):
            sentences = example.get("sentences", [])

            if not sentences:
                skipped_empty += 1
                continue

            full_text = " ".join(sentences)

            encoding = self.tokenizer(
                full_text,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_special_tokens_mask=True,
            )

            # Sample ONE random window (or take all if shorter than max_length)
            if len(encoding["input_ids"]) > self.max_length:
                examples_sampled += 1
                max_start = len(encoding["input_ids"]) - self.max_length
                start_idx = random.randint(0, max_start)
                chunk_ids = encoding["input_ids"][start_idx:start_idx + self.max_length]
                chunk_mask = encoding["attention_mask"][start_idx:start_idx + self.max_length]
            else:
                examples_within_max += 1
                chunk_ids = encoding["input_ids"]
                chunk_mask = encoding["attention_mask"]

            processed.append({
                "input_ids": chunk_ids,
                "attention_mask": chunk_mask,
                "sic2": example.get("sic2"),
                "sic3": example.get("sic3"),
                "sic4": example.get("sic4"),
            })

        print(f"\n{'='*80}")
        print(f"DEBUG: Preprocessing Statistics")
        print(f"{'='*80}")
        print(f"Skipped (empty sentences): {skipped_empty}")
        print(f"Examples within max_length: {examples_within_max}")
        print(f"Examples requiring sampling: {examples_sampled}")
        print(f"Total processed examples: {len(processed)}")
        print(f"{'='*80}\n")

        return processed


    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]

        sic2 = self._map_raw_sic_code_to_index(example["sic2"], self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(example["sic3"], self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(example["sic4"], self.indexed_sic4_list)

        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }


    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100)


    def __len__(self) -> int:
        return len(self.examples)


class PretrainDatasetOnTheFly(Dataset):
    """
    Pretraining dataset with ON-THE-FLY SAMPLING strategy.

    Instead of preprocessing, tokenizes and samples during __getitem__.
    """
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
        )

        # Sample random window
        if len(encoding["input_ids"]) > self.max_length:
            max_start = len(encoding["input_ids"]) - self.max_length
            start_idx = random.randint(0, max_start)
            input_ids = encoding["input_ids"][start_idx:start_idx + self.max_length]
            attention_mask = encoding["attention_mask"][start_idx:start_idx + self.max_length]
        else:
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

        sic2 = self._map_raw_sic_code_to_index(example.get("sic2"), self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(example.get("sic3"), self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(example.get("sic4"), self.indexed_sic4_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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

