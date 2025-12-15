# BusinessBERT2

This repository contains the code used for the pretraining of BusinessBERT2. It is a domain-adapted BERT-style encoder for business communication.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install hf_transfer
pip install wandb

wandb login

nohup python -m src.training.pretrain \
	--config config/pretrain.yaml \
	--data ../../data/pretraining_dataset.jsonl \
	--report_to wandb \
	--max_seq_length 512 \
	--batch_size 64 \
	--learning_rate 5e-5 \
	--lr_scheduler_type linear \
	--num_train_epochs 1 \
	--max_steps 1000000 \
	--warmup_steps 10000 \
	--num_workers 16 \
	--weight_decay 0.01 \
	--gradient_accumulation_steps 2 \
	--grad_clip 1.0 \
	--precision bf16 \
	--logging_steps 500 \
	--eval_strategy steps \
	--eval_steps 2000 \
	--save_total_limit 3 \
	--save_strategy steps \
	--save_steps 2000 \
	--load_best_model_at_end True \
	--metric_for_best_model eval_loss \
	--greater_is_better False \
	--val_ratio 0.1 \
	--wandb_mode online \
	--wandb_project businessbert2-pretraining \
	--save_dir ../../outputs/output-path \
	--save_safetensors False \
	--safe_serialization False \
	> businessbert2-pretraining.log 2>&1 &
