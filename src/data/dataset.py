from typing import Dict, Any, List, Optional
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
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4
        self.nsp_probability = nsp_probability

    def __len__(self):
        return len(self.raw_examples)

    def _map_sic_code(self, sic_value: Optional[str], sic_to_idx: Dict[str, int]) -> int:
        """Map SIC code to index, handling None and 'NA' values."""
        if sic_value is None or sic_value == "NA" or sic_value == "":
            return -100
        return sic_to_idx.get(str(sic_value), -100)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.raw_examples[idx]
        raw_sentences = example["sentences"]
        sentences = ' '.join(raw_sentences)

        # if len(sentences) == 0:
        #     sentences = [""]
        #
        # # Split document roughly in half to create two segments
        # # Each segment contains multiple consecutive sentences
        # split_point = len(sentences) // 2
        # if split_point == 0:
        #     split_point = 1
        #
        # # Segment A: first half of document
        # segment_a_sentences = sentences[:split_point]
        #
        # # NSP logic
        # if random.random() < self.nsp_probability:
        #     # Segment B: random sentences from different document (label = 1)
        #     random_idx = random.randint(0, len(self.raw_examples) - 1)
        #     random_sentences = self.raw_examples[random_idx]["sentences"]
        #     # Take some sentences from random document
        #     num_sentences = min(len(random_sentences), split_point)
        #     segment_b_sentences = random_sentences[:num_sentences] if num_sentences > 0 else [""]
        #     nsp_label = 1
        # else:
        #     # Segment B: second half of same document (label = 0)
        #     segment_b_sentences = sentences[split_point:] if split_point < len(sentences) else [""]
        #     nsp_label = 0
        #
        # # Join sentences into two text segments
        # segment_a = " ".join(segment_a_sentences)
        # segment_b = " ".join(segment_b_sentences)

        # Tokenize the pair
        encoding = self.tokenizer(
            sentences,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        # Map SIC codes
        sic2 = self._map_sic_code(example.get("sic2"), self.idx2)
        sic3 = self._map_sic_code(example.get("sic3"), self.idx3)
        sic4 = self._map_sic_code(example.get("sic4"), self.idx4)

        return {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding["token_type_ids"],
            "attention_mask": encoding["attention_mask"],
            "nsp_label": 0,
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }