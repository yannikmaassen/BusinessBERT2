import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetWithNSPSegments(Dataset):
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

        # We keep only docs with >= 2 sentences (as you do now)
        self.valid_examples = []
        for example in tqdm(raw_examples, desc="Filtering valid examples"):
            sentences = example.get("sentences", [])
            if len(sentences) >= 2:
                self.valid_examples.append(example)

        # How many tokens we allow per segment (rough split)
        # Reserve 3 for [CLS], [SEP], [SEP]
        self.max_total_tokens = max_length - 3
        self.max_segment_tokens = self.max_total_tokens // 2

    # ---- helper to build a segment from multiple sentences ----
    def _build_segment(self, sentences, start_idx, max_tokens):
        """
        Greedily append sentences starting at start_idx
        until token budget is reached.
        Returns the text span and the index of the next sentence.
        """
        current = []
        for i in range(start_idx, len(sentences)):
            current.append(sentences[i])
            # tokenize without special tokens just to count
            enc = self.tokenizer(
                " ".join(current),
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
            if len(enc["input_ids"]) >= max_tokens:
                # last sentence overflowed -> drop it and stop
                current.pop()
                break

        if not current:
            # fallback: at least use one sentence
            current = [sentences[start_idx]]
            next_idx = min(start_idx + 1, len(sentences))
        else:
            next_idx = start_idx + len(current)

        span_text = " ".join(current)
        return span_text, next_idx

    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.valid_examples[idx]
        sentences = example.get("sentences", [])

        # --- NSP label ---
        is_next = random.random() < 0.5

        # ---- build segment A from this document ----
        # ensure there is at least one sentence left for segment B if is_next
        if is_next and len(sentences) >= 2:
            max_first_idx = len(sentences) - 1  # need at least one after
        else:
            max_first_idx = len(sentences)  # any start is fine

        first_idx = random.randint(0, max_first_idx - 1)
        seg_a, next_idx = self._build_segment(
            sentences,
            start_idx=first_idx,
            max_tokens=self.max_segment_tokens,
        )

        # ---- build segment B ----
        if is_next:
            # take following sentences from same document
            if next_idx >= len(sentences):
                # if span A already used up the document, fall back to a small last part
                next_idx = max(first_idx + 1, len(sentences) - 1)
            seg_b, _ = self._build_segment(
                sentences,
                start_idx=next_idx,
                max_tokens=self.max_segment_tokens,
            )
            nsp_label = 0
        else:
            # random span from a random *other* document
            rand_example = random.choice(self.valid_examples)
            rand_sentences = rand_example.get("sentences", [])
            rand_start = random.randint(0, len(rand_sentences) - 1)
            seg_b, _ = self._build_segment(
                rand_sentences,
                start_idx=rand_start,
                max_tokens=self.max_segment_tokens,
            )
            nsp_label = 1

        # ---- final tokenization of (A, B) pair ----
        encoding = self.tokenizer(
            seg_a,
            seg_b,
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
