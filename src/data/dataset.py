from typing import Dict, Any, List, Optional, Tuple
import random
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset):
    def __init__(
            self,
            raw_examples: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizerBase,
            max_length: int,
            idx2: Dict[str, int],
            idx3: Dict[str, int],
            idx4: Dict[str, int],
            nsp_probability: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4

        print("Preprocessing segments...")
        self.examples = self._preprocess_examples(raw_examples, nsp_probability)
        print(f"Created {len(self.examples)} training examples")

    def _preprocess_examples(
            self,
            raw_examples: List[Dict[str, Any]],
            nsp_probability: float
    ) -> List[Dict[str, Any]]:
        """Preprocess all examples once during initialization."""
        processed = []

        # Build index of valid documents
        valid_indices = [
            i for i, ex in enumerate(raw_examples)
            if ex.get("sentences") and len(ex["sentences"]) > 0
        ]

        for idx, example in enumerate(raw_examples):
            sentences = example.get("sentences", [])

            if not sentences:
                continue

            # Calculate split point
            split_point = max(1, len(sentences) // 2)
            segment_a_sentences = sentences[:split_point]

            # Decide NSP label
            if random.random() < nsp_probability and len(valid_indices) > 1:
                # Negative: random document
                random_idx = random.choice([i for i in valid_indices if i != idx])
                random_sentences = raw_examples[random_idx]["sentences"]

                target_length = len(sentences) - split_point
                num_sentences = min(len(random_sentences), max(1, target_length))

                if len(random_sentences) > num_sentences:
                    start_idx = random.randint(0, len(random_sentences) - num_sentences)
                    segment_b_sentences = random_sentences[start_idx:start_idx + num_sentences]
                else:
                    segment_b_sentences = random_sentences

                nsp_label = 1
            else:
                # Positive: same document
                segment_b_sentences = sentences[split_point:] if split_point < len(sentences) else [sentences[-1]]
                nsp_label = 0

            # Tokenize immediately
            segment_a = " ".join(segment_a_sentences)
            segment_b = " ".join(segment_b_sentences)

            encoding = self.tokenizer(
                segment_a,
                segment_b,
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Don't pad during preprocessing
                return_tensors=None,
            )

            processed.append({
                "input_ids": encoding["input_ids"],
                "token_type_ids": encoding["token_type_ids"],
                "attention_mask": encoding["attention_mask"],
                "nsp_label": nsp_label,
                "sic2": example.get("sic2"),
                "sic3": example.get("sic3"),
                "sic4": example.get("sic4"),
            })

        return processed

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]

        # Map SIC codes (very fast)
        sic2 = self._map_sic_code(example["sic2"], self.idx2)
        sic3 = self._map_sic_code(example["sic3"], self.idx3)
        sic4 = self._map_sic_code(example["sic4"], self.idx4)

        return {
            "input_ids": example["input_ids"],
            "token_type_ids": example["token_type_ids"],
            "attention_mask": example["attention_mask"],
            "nsp_label": example["nsp_label"],
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }
