import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetWithNSP(Dataset):
    """
    Dataset that creates sentence pairs for NSP on-the-fly.
    50% of pairs are consecutive (label=0), 50% are random (label=1).
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

        # Filter valid examples
        self.valid_examples = []
        for example in tqdm(raw_examples, desc="Filtering valid examples"):
            sentences = example.get("sentences", [])
            if len(sentences) >= 2:  # Need at least 2 sentences for NSP
                self.valid_examples.append(example)

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.valid_examples[idx]
        sentences = example.get("sentences", [])

        # NSP: 50% consecutive, 50% random
        is_next = random.random() < 0.5

        if len(sentences) >= 2:
            # Pick first sentence randomly
            max_first_idx = len(sentences) - 1 if is_next else len(sentences)
            first_idx = random.randint(0, max_first_idx - 1)

            sent_a = sentences[first_idx]

            if is_next:
                # Take next sentence
                sent_b = sentences[first_idx + 1]
                nsp_label = 0
            else:
                # Take random sentence from random document
                random_example = random.choice(self.valid_examples)
                random_sentences = random_example.get("sentences", [])
                sent_b = random.choice(random_sentences)
                nsp_label = 1
        else:
            # Fallback: use whole text
            sent_a = " ".join(sentences[:len(sentences) // 2])
            sent_b = " ".join(sentences[len(sentences) // 2:])
            nsp_label = 0

        # Tokenize with segment IDs
        encoding = self.tokenizer(
            sent_a,
            sent_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        sic2 = self._map_raw_sic_code_to_index(example.get("sic2"), self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(example.get("sic3"), self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(example.get("sic4"), self.indexed_sic4_list)

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
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
