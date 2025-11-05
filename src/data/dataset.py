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

        # DEBUG: Print raw examples info
        print(f"\n{'='*30}")
        print(f"DEBUG: Dataset Initialization")
        print(f"{'='*30}")
        print(f"Total raw examples: {len(raw_examples)}")
        print(f"Max length: {max_length}")
        if raw_examples:
            print(f"\nSample raw example (first):")
            sample = raw_examples[0]
            print(f"  SIC2: {sample.get('sic2')}")
            print(f"  SIC3: {sample.get('sic3')}")
            print(f"  SIC4: {sample.get('sic4')}")
            print(f"  Number of sentences: {len(sample.get('sentences', []))}")
            sentences = sample.get('sentences', [])
            if sentences:
                print(f"  First sentence preview: {sentences[0][:100]}...")
        print(f"{'='*30}\n")

        print("Preprocessing examples...")
        self.examples = self._preprocess_examples(raw_examples[0:1000])
        print(f"Created {len(self.examples)} training examples")

        # DEBUG: Print processed examples info
        print(f"\n{'='*30}")
        print(f"DEBUG: After Preprocessing")
        print(f"{'='*30}")
        if self.examples:
            print(f"Sample processed example (first):")
            sample_processed = self.examples[0]
            print(f"  Input IDs length: {len(sample_processed['input_ids'])}")
            print(f"  Input IDs (first 20): {sample_processed['input_ids'][:20]}")
            print(f"  Attention mask length: {len(sample_processed['attention_mask'])}")
            print(f"  SIC2: {sample_processed.get('sic2')}")
            print(f"  SIC3: {sample_processed.get('sic3')}")
            print(f"  SIC4: {sample_processed.get('sic4')}")
            print(f"  Decoded text preview: {self.tokenizer.decode(sample_processed['input_ids'][:50])}...")
        print(f"{'='*30}\n")

        # Move tokenizer back to original device if needed
        if original_device is not None and hasattr(self.tokenizer, 'to'):
            self.tokenizer = self.tokenizer.to(original_device)


    def __getitem__(self, idx) -> Dict[str, Any]:
        example = self.examples[idx]

        sic2 = self._map_raw_sic_code_to_index(example["sic2"], self.indexed_sic2_list)
        sic3 = self._map_raw_sic_code_to_index(example["sic3"], self.indexed_sic3_list)
        sic4 = self._map_raw_sic_code_to_index(example["sic4"], self.indexed_sic4_list)

        # DEBUG: Print first few accesses
        if idx < 3:
            print(f"\n--- DEBUG __getitem__ idx={idx} ---")
            print(f"Input IDs length: {len(example['input_ids'])}")
            print(f"Raw SIC2: {example['sic2']} -> Index: {sic2}")
            print(f"Raw SIC3: {example['sic3']} -> Index: {sic3}")
            print(f"Raw SIC4: {example['sic4']} -> Index: {sic4}")
            print(f"Decoded text (first 100 chars): {self.tokenizer.decode(example['input_ids'][:20])}...")

        # sic2, sic3 and sic4 are now returned as indices rather than actual SIC codes
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "sic2": sic2,
            "sic3": sic3,
            "sic4": sic4,
        }


    def _preprocess_examples(self, raw_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []

        # DEBUG: Track statistics
        skipped_empty = 0
        examples_within_max = 0
        examples_chunked = 0
        chunks_created = 0
        chunks_skipped_too_short = 0

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

            # DEBUG: Print detailed info for first few examples
            if idx < 3:
                print(f"\n--- DEBUG Example {idx} ---")
                print(f"Text length (chars): {len(full_text)}")
                print(f"Token length: {len(encoding['input_ids'])}")
                print(f"Will chunk: {len(encoding['input_ids']) > self.max_length}")

            # If within max_length, process normally
            if len(encoding["input_ids"]) <= self.max_length:
                examples_within_max += 1
                processed.append({
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "sic2": example.get("sic2"),
                    "sic3": example.get("sic3"),
                    "sic4": example.get("sic4"),
                })

                if idx < 3:
                    print(f"Added as single example (length: {len(encoding['input_ids'])})")
            else:
                examples_chunked += 1
                chunk_count = 0
                for i in range(0, len(encoding["input_ids"]), self.max_length):
                    chunk_ids = encoding["input_ids"][i:i + self.max_length]
                    chunk_mask = encoding["attention_mask"][i:i + self.max_length]

                    if len(chunk_ids) < 64:
                        chunks_skipped_too_short += 1
                        if idx < 3:
                            print(f"Skipped chunk (too short: {len(chunk_ids)})")
                        continue

                    chunks_created += 1
                    chunk_count += 1
                    processed.append({
                        "input_ids": chunk_ids,
                        "attention_mask": chunk_mask,
                        "sic2": example.get("sic2"),
                        "sic3": example.get("sic3"),
                        "sic4": example.get("sic4"),
                    })

                if idx < 3:
                    print(f"Chunked into {chunk_count} chunks")

        # DEBUG: Print summary statistics
        print(f"\n{'='*30}")
        print(f"DEBUG: Preprocessing Statistics")
        print(f"{'='*30}")
        print(f"Skipped (empty sentences): {skipped_empty}")
        print(f"Examples within max_length: {examples_within_max}")
        print(f"Examples requiring chunking: {examples_chunked}")
        print(f"Total chunks created from long examples: {chunks_created}")
        print(f"Chunks skipped (< 64 tokens): {chunks_skipped_too_short}")
        print(f"Total processed examples: {len(processed)}")
        print(f"{'='*30}\n")

        return processed


    def _map_raw_sic_code_to_index(self, sic_code: Optional[str], indexed_sic_list: Dict[str, int]) -> int:
        if sic_code is None or sic_code == "NA" or sic_code == "":
            return -100
        return indexed_sic_list.get(str(sic_code), -100) # for key=raw SIC code get corresponding index or -100 if not found


    def __len__(self) -> int:
        return len(self.examples)
