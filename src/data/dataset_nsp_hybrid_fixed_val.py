import random
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PretrainDatasetWithNSPOptimizedFixedEval(Dataset):
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

        self.max_total_tokens = max_length - 3
        self.per_segment_budget = self.max_total_tokens // 2

        self.valid_examples = [
            ex for ex in raw_examples if ex.get("sentences")
        ]

        if not is_training:
            self._prebuilt_examples = self._prebuild_validation_examples(seed)


    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.is_training:
            # Training: random sampling each time
            return self._get_random_example(idx)
        else:
            # Validation: return pre-built example
            return self._get_prebuilt_example(idx)


    def _prebuild_validation_examples(self, seed: int) -> List[Dict[str, Any]]:
        """Pre-build validation examples with fixed random seed."""
        rng = random.Random(seed)
        prebuilt = []

        for example_a in tqdm(self.valid_examples, desc="Pre-building validation examples"):
            sentences = example_a.get("sentences", [])

            # Use fixed RNG for validation
            is_next = rng.random() < 0.5

            if is_next:
                max_start = max(0, len(sentences) - 2)
                start_idx = rng.randint(0, max_start)

                a_ids, next_idx = self._build_sentence_aware_segment(
                    sentences, start_idx, self.per_segment_budget
                )

                remaining_budget = self.max_total_tokens - len(a_ids)

                if next_idx >= len(sentences):
                    next_idx = max(start_idx + 1, len(sentences) - 1)

                b_ids = self._fill_remaining_tokens(
                    sentences, next_idx, remaining_budget
                )

                nsp_label = 0
                meta_example = example_a
            else:
                start_idx = rng.randint(0, len(sentences) - 1)

                a_ids, _ = self._build_sentence_aware_segment(
                    sentences, start_idx, self.per_segment_budget
                )

                remaining_budget = self.max_total_tokens - len(a_ids)

                example_b = rng.choice(self.valid_examples)
                while example_b is example_a and len(self.valid_examples) > 1:
                    example_b = rng.choice(self.valid_examples)

                b_sentences = example_b.get("sentences", [])
                b_start = rng.randint(0, len(b_sentences) - 1)

                b_ids = self._fill_remaining_tokens(
                    b_sentences, b_start, remaining_budget
                )

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

        # Padding
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
        example_a = self.valid_examples[idx]
        sentences = example_a.get("sentences", [])

        is_next = random.random() < 0.5

        if is_next:
            # Positive: segments from same document
            max_start = max(0, len(sentences) - 2)
            start_idx = random.randint(0, max_start)

            # Build segment A with complete sentences
            a_ids, next_idx = self._build_sentence_aware_segment(
                sentences, start_idx, self.per_segment_budget
            )

            # Calculate remaining budget (total budget minus what A used)
            remaining_budget = self.max_total_tokens - len(a_ids)

            # Fill segment B with tokens (ignore sentence boundaries)
            if next_idx >= len(sentences):
                next_idx = max(start_idx + 1, len(sentences) - 1)

            b_ids = self._fill_remaining_tokens(
                sentences, next_idx, remaining_budget
            )

            nsp_label = 0
            meta_example = example_a

        else:
            # Negative: A from current doc, B from different doc
            start_idx = random.randint(0, len(sentences) - 1)

            # Build segment A with complete sentences
            a_ids, _ = self._build_sentence_aware_segment(
                sentences, start_idx, self.per_segment_budget
            )

            # Calculate remaining budget
            remaining_budget = self.max_total_tokens - len(a_ids)

            # Get segment B from different document
            example_b = random.choice(self.valid_examples)
            while example_b is example_a and len(self.valid_examples) > 1:
                example_b = random.choice(self.valid_examples)

            b_sentences = example_b.get("sentences", [])
            b_start = random.randint(0, len(b_sentences) - 1)

            # Fill segment B with tokens (ignore sentence boundaries)
            b_ids = self._fill_remaining_tokens(
                b_sentences, b_start, remaining_budget
            )

            nsp_label = 1
            meta_example = example_a

        # Build final inputs with special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(a_ids, b_ids)
        attention_mask = [1] * len(input_ids)

        # Minimal padding (should be very little now)
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

    def _build_sentence_aware_segment(
            self,
            sentences: List[str],
            start_idx: int,
            max_tokens: int
    ) -> tuple[List[int], int]:
        """
        Accumulate complete sentences up to the token budget.
        Returns token IDs and next sentence index.
        """
        accumulated_ids = []
        current_idx = start_idx

        for i in range(start_idx, len(sentences)):
            sent_ids = self.tokenizer(
                sentences[i],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            # Check if adding this sentence exceeds budget
            if len(accumulated_ids) + len(sent_ids) > max_tokens:
                # Don't add this sentence, stop here
                break

            # Add this complete sentence
            accumulated_ids.extend(sent_ids)
            current_idx = i + 1

        # Fallback: ensure we have at least one sentence (even if truncated)
        if not accumulated_ids and start_idx < len(sentences):
            sent_ids = self.tokenizer(
                sentences[start_idx],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            accumulated_ids = sent_ids[:max_tokens]
            current_idx = start_idx + 1

        return accumulated_ids, current_idx

    def _fill_remaining_tokens(
            self,
            sentences: List[str],
            start_idx: int,
            remaining_tokens: int
    ) -> List[int]:
        """
        Fill remaining token budget by tokenizing from start_idx onwards,
        ignoring sentence boundaries. May cut off mid-sentence.
        """
        # Join all remaining sentences into one text
        remaining_text = " ".join(sentences[start_idx:])

        # Tokenize and take exactly what we need
        token_ids = self.tokenizer(
            remaining_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Take exactly the remaining budget (may truncate mid-sentence)
        return token_ids[:remaining_tokens]

    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100)

    def __len__(self) -> int:
        return len(self.valid_examples)
