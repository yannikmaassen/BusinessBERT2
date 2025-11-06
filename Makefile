# ====== Settings ======
VENV?=.venv
PY?=python
PIP?=pip
CONFIG?=config/pretrain.yaml

.PHONY: help venv install install-colab train train-data train-colab train-data-colab test clean pretrain

help:
	@echo "Common commands:"
	@echo "  make venv                 # create local virtualenv"
	@echo "  make install              # install deps into venv"
	@echo "  make install-colab        # install deps in current environment (Colab)"
	@echo "  make train                # run smoke pretrain with $(CONFIG) (uses venv)"
	@echo "  make train-data DATA=...  # same but override dataset path"
	@echo "  make train-colab          # run pretrain in current env (Colab)"
	@echo "  make train-data-colab DATA=...   # pretrain w/ explicit data path (Colab)"
	@echo "  make test                 # quick import test"
	@echo "  make clean                # remove caches / outputs"

# ====== Local dev (venv) ======
venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip

install:
	. $(VENV)/bin/activate; $(PIP) install --no-cache-dir -r requirements.txt

train:
	. $(VENV)/bin/activate; $(PY) -m src.training.pretrain --config $(CONFIG)

train-data:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-data DATA=/path/to/file.jsonl"; exit 1; fi
	. $(VENV)/bin/activate; $(PY) -m src.training.pretrain --config $(CONFIG) --data $(DATA)

test:
	. $(VENV)/bin/activate; $(PY) -c "import transformers, torch; print('OK', transformers.__version__)"

# ====== Colab / any pre-provisioned env (no venv) ======
install-colab:
	$(PIP) -q install --no-cache-dir -r requirements.txt

train-colab:
	$(PY) -m src.training.pretrain --config $(CONFIG)

train-data-colab:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl"; exit 1; fi
	$(PY) -m src.training.pretrain \
		--config $(CONFIG) \
		--data $(DATA) \
		--report_to $(REPORT_TO) \
		--max_seq_length $(MAX_SEQ_LENGTH) \
		--batch_size $(BATCH_SIZE) \
		--learning_rate $(LEARNING_RATE) \
		--num_train_epochs $(NUM_TRAIN_EPOCHS) \
		--max_steps $(MAX_STEPS) \
		--warmup_steps $(WARMUP_STEPS) \
		--num_workers $(NUM_WORKERS) \
		--weight_decay $(WEIGHT_DECAY) \
		--gradient_accumulation_steps $(GRADIENT_ACCUMULATION_STEPS) \
		--grad_clip $(GRAD_CLIP) \
		--precision $(PRECISION) \
		--logging_steps $(LOGGING_STEPS) \
		--eval_strategy $(EVAL_STRATEGY) \
		--eval_steps $(EVAL_STEPS) \
		--save_total_limit $(SAVE_TOTAL_LIMIT) \
		--save_strategy $(SAVE_STRATEGY) \
		--save_steps $(SAVE_STEPS) \
		--load_best_model_at_end $(LOAD_BEST_MODEL_AT_END) \
		--metric_for_best_model $(METRIC_FOR_BEST_MODEL) \
		--greater_is_better $(GREATER_IS_BETTER) \
		--val_ratio $(VAL_RATIO)

# ====== Cleanup ======
clean:
	rm -rf __pycache__ .pytest_cache */__pycache__ *.egg-info
	rm -rf outputs checkpoints

pretrain:
	python -m src.training.pretrain --config config/pretrain.yaml --data ./data/sample.jsonl

analyze_vocab:
	python -m src.analyze_vocab --config config/pretrain.yaml --data ./data/sample.jsonl

analyze-vocab-colab:
	$(PY) -m src.analyze_vocab --data $(DATA)