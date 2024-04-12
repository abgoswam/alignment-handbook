

#!/bin/bash

# Step 1: Output of pip list
pip list


# Hardcoded values
INPUT_USE_FLASH_ATTENTION="true"
INPUT_MAX_LENGTH=131072
INPUT_MAX_POSITION_EMBEDDINGS=131072
INPUT_TRIALS=1
INPUT_SEED=98052

# INPUT_MODEL_PATH="phi_7B_phase2_iter_165462_20240401"
# INPUT_MODEL_PATH="phi37b_rc1_0_rope1_2_tfm439_20240404"
INPUT_MODEL_PATH="phi3_phase2v1_1_3_8b_phase2_20240325_rc1_0"

OUTPUT_OUTPUT_DIR="."

# Step 2: Run the Python script with provided inputs
CUDA_VISIBLE_DEVICES=1 python eval_passkey_nikosk.py \
    --use_flash_attention "${INPUT_USE_FLASH_ATTENTION}" \
    --max_length "${INPUT_MAX_LENGTH}" \
    --max_position_embeddings "${INPUT_MAX_POSITION_EMBEDDINGS}" \
    --trials "${INPUT_TRIALS}" \
    --seed "${INPUT_SEED}" \
    --model_path "${INPUT_MODEL_PATH}" \
    --output_dir "${OUTPUT_OUTPUT_DIR}"


# Output "DONE"
echo "DONE"