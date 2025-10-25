from typing import Dict, Any, List, Optional
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
    ):
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4

        # Build chunk index for efficient sampling
        self.chunk_index = self._build_chunk_index()
        print(f"Dataset initialized with {len(self.raw_examples)} documents, {len(self.chunk_index)} chunks")

    def _build_chunk_index(self) -> List[Dict[str, Any]]:
        """Build index of all chunks without tokenizing."""
        chunk_index = []

        for doc_idx, example in enumerate(self.raw_examples):
            sentences = example.get("sentences", [])
            if not sentences:
                continue

            # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
            full_text = " ".join(sentences)
            estimated_tokens = len(full_text) // 4

            if estimated_tokens <= self.max_length:
                # Single chunk
                chunk_index.append({
                    "doc_idx": doc_idx,
                    "chunk_start": 0,
                    "chunk_end": None,  # Full document
                })
            else:
                # Multiple chunks with stride
                stride = int(self.max_length * 0.95)
                for start in range(0, estimated_tokens, stride):
                    chunk_index.append({
                        "doc_idx": doc_idx,
                        "chunk_start": start,
                        "chunk_end": start + self.max_length,
                    })

        return chunk_index

    def __getitem__(self, idx) -> Dict[str, Any]:
        chunk_info = self.chunk_index[idx]
        example = self.raw_examples[chunk_info["doc_idx"]]

        # Get full text
        sentences = example.get("sentences", [])
        full_text = " ".join(sentences)

        # Tokenize full document
        encoding = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        # Extract chunk
        start = chunk_info["chunk_start"]
        end = chunk_info["chunk_end"]

        if end is None:
            # Full document fits in max_length
            chunk_ids = encoding["input_ids"]
            chunk_mask = encoding["attention_mask"]
        else:
            # Extract chunk with actual token positions
            chunk_ids = encoding["input_ids"][start:end]
            chunk_mask = encoding["attention_mask"][start:end]

        # Pad to max_length
        padding_length = self.max_length - len(chunk_ids)
        if padding_length > 0:
            chunk_ids = chunk_ids + [self.tokenizer.pad_token_id] * padding_length
            chunk_mask = chunk_mask + [0] * padding_length
        else:
            # Truncate if needed
            chunk_ids = chunk_ids[:self.max_length]
            chunk_mask = chunk_mask[:self.max_length]

        return {
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask,
            "sic2": self._map_sic_code(example.get("sic2"), self.idx2),
            "sic3": self._map_sic_code(example.get("sic3"), self.idx3),
            "sic4": self._map_sic_code(example.get("sic4"), self.idx4),
        }

    def _map_sic_code(self, sic_value: Optional[str], sic_to_idx: Dict[str, int]) -> int:
        """Map SIC code to index, handling None and 'NA' values."""
        if sic_value is None or sic_value == "NA" or sic_value == "":
            return -100
        return sic_to_idx.get(str(sic_value), -100)

    def __len__(self) -> int:
        return len(self.chunk_index)
