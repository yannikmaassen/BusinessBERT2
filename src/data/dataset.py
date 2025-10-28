from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDataset(Dataset):
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
        original_device = next(self.tokenizer.parameters()).device if hasattr(self.tokenizer, 'parameters') else None

        # Temporarily move tokenizer to CPU for preprocessing
        if hasattr(self.tokenizer, 'to'):
            self.tokenizer = self.tokenizer.to(preprocess_device)

        print("Preprocessing examples...")
        self.examples = self._preprocess_examples(raw_examples[0:1000])
        print(f"Created {len(self.examples)} training examples")

        # Move tokenizer back to original device if needed
        if original_device is not None and hasattr(self.tokenizer, 'to'):
            self.tokenizer = self.tokenizer.to(original_device)


    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]

        # sic2, sic3 and sic4 are now returned as indices rather than actual SIC codes
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "token_type_ids": example["token_type_ids"],
            "sic2": self._map_raw_sic_code_to_index(example["sic2"], self.indexed_sic2_list),
            "sic3": self._map_raw_sic_code_to_index(example["sic3"], self.indexed_sic3_list),
            "sic4": self._map_raw_sic_code_to_index(example["sic4"], self.indexed_sic4_list),
        }


    def _preprocess_examples(self, raw_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []

        for example in tqdm(raw_examples, desc="Preprocessing examples"):
            sentences = example.get("sentences", [])

            if not sentences:
                continue

            full_text = " ".join(sentences)

            encoding = self.tokenizer(
                full_text,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_special_tokens_mask=True,
            )

            # If within max_length, process normally
            if len(encoding["input_ids"]) <= self.max_length:
                processed.append({
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "token_type_ids": encoding["token_type_ids"],
                    "sic2": example.get("sic2"),
                    "sic3": example.get("sic3"),
                    "sic4": example.get("sic4"),
                })
            else:
                for i in range(0, len(encoding["input_ids"]), self.max_length):
                    chunk_ids = encoding["input_ids"][i:i + self.max_length]
                    chunk_mask = encoding["attention_mask"][i:i + self.max_length]
                    chunk_types = encoding["token_type_ids"][i:i + self.max_length]

                    if len(chunk_ids) < 64:
                        continue

                    processed.append({
                        "input_ids": chunk_ids,
                        "attention_mask": chunk_mask,
                        "token_type_ids": chunk_types,
                        "sic2": example.get("sic2"),
                        "sic3": example.get("sic3"),
                        "sic4": example.get("sic4"),
                    })

        return processed


    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100) # for key=raw SIC code get corresponding index or -100 if not found


    def __len__(self) -> int:
        return len(self.examples)
