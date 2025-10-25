from typing import Dict, Any, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm


class PretrainDataset(Dataset):
    def __init__(
            self,
            raw_examples: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizerBase,
            max_length: int,
            idx2: Dict[str, int],
            idx3: Dict[str, int],
            idx4: Dict[str, int],
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4

        print("Preprocessing examples...")
        self.examples = self._preprocess_examples(raw_examples)
        print(f"Created {len(self.examples)} training examples")


    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]

        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "sic2": self._map_sic_code(example["sic2"], self.idx2),
            "sic3": self._map_sic_code(example["sic3"], self.idx3),
            "sic4": self._map_sic_code(example["sic4"], self.idx4),
        }


    def _preprocess_examples(
            self,
            raw_examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Preprocess all examples - document level for MLM + IC."""
        processed = []

        for example in tqdm(raw_examples, desc="Preprocessing examples"):
            sentences = example.get("sentences", [])

            if not sentences:
                continue

            # Full document as single segment
            full_text = " ".join(sentences)

            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )

            processed.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "sic2": example.get("sic2"),
                "sic3": example.get("sic3"),
                "sic4": example.get("sic4"),
            })

        return processed


    def _map_sic_code(self, code: Any, mapping: Dict[str, int]) -> int:
        if code is None:
            return -1
        return mapping.get(str(code), -1)


    def __len__(self) -> int:
        return len(self.examples)
