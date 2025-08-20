# BusinessBERT2

This repo holds a minimal, reproducible scaffold to develop and run BusinessBERT 2.0 pretraining from a GitHub repo while executing on Colab/GPU or a remote VM.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.businessbert2.training.pretrain --config configs/pretrain.yaml --data ./data/sample.jsonl