from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling
import torch


@dataclass
class DataCollatorForPretraining(DataCollatorForLanguageModeling):
    """Handles MLM and preserves SIC metadata (no NSP)."""

    def __call__(self, features):
        # Extract SIC codes before MLM processing
        sic2 = torch.tensor([f.pop("sic2") for f in features])
        sic3 = torch.tensor([f.pop("sic3") for f in features])
        sic4 = torch.tensor([f.pop("sic4") for f in features])

        # Apply MLM masking
        batch = super().__call__(features)

        # Add back SIC data
        batch["sic2"] = sic2
        batch["sic3"] = sic3
        batch["sic4"] = sic4

        return batch
