import os
import re

CHECKPOINT_PREFIX = "checkpoint-"

def find_latest_checkpoint(output_dir: str):
    """
    Return path to latest checkpoint directory or None.
    A valid checkpoint must contain:
      - model file (pytorch_model.bin or model.safetensors)
      - optimizer.pt
      - trainer_state.json
    """
    if not os.path.isdir(output_dir):
        return None

    latest_step = -1
    latest_path = None
    pattern = re.compile(rf"^{CHECKPOINT_PREFIX}(\d+)$")

    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        m = pattern.match(name)
        if not m:
            continue
        step = int(m.group(1))

        # Validate expected files
        has_model = (
            os.path.isfile(os.path.join(full, "pytorch_model.bin")) or
            os.path.isfile(os.path.join(full, "model.safetensors"))
        )
        has_opt = os.path.isfile(os.path.join(full, "optimizer.pt"))
        has_state = os.path.isfile(os.path.join(full, "trainer_state.json"))
        if not (has_model and has_opt and has_state):
            continue

        if step > latest_step:
            latest_step = step
            latest_path = full

    return latest_path