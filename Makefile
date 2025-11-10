# ====== Settings ======
PY?=python
PIP?=pip
CONFIG?=config/pretrain.yaml

.PHONY: help venv install install-colab train train-data train-colab train-data-colab test clean pretrain

help:
	@echo "Common commands:"
	@echo "  make install-colab        # install deps in current environment (Colab)"
	@echo "  make train-colab          # run pretrain in current env (Colab)"
	@echo "  make train-data-colab DATA=...   # pretrain w/ explicit data path (Colab)"

# ====== Colab / any pre-provisioned env (no venv) ======
install-colab:
	$(PIP) -q install --no-cache-dir -r requirements.txt

train-colab:
	$(PY) -m src.training.pretrain --config $(CONFIG)

train-data-colab:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-data-colab DATA=/path/to/data"; exit 1; fi
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
		--val_ratio $(VAL_RATIO) \
		--wandb_mode $(WANDB_MODE) \
		--wandb_project $(WANDB_PROJECT) \
		--save_dir $(SAVE_DIR) \
		--save_safetensors $(SAVE_SAFETENSORS) \
		--safe_serialization $(SAFE_SERIALIZATION)