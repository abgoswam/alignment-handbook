

#!/bin/bash

# Step 1: Output of pip list
pip list


# Hardcoded values
INPUT_MAX_LENGTH=131072
INPUT_TRIALS=50
INPUT_ROPE_SCALING_FACTOR=1.0
INPUT_ROPE_SCALING_TYPE="linear"
INPUT_SEED=98052
INPUT_MODEL_PATH="phi_7B_phase2_iter_165462_20240401"
OUTPUT_OUTPUT_DIR="."

# Step 2: Run the Python script with provided inputs
python eval_passkey.py \
    --max_length "${INPUT_MAX_LENGTH}" \
    --trials "${INPUT_TRIALS}" \
    --rope_scaling_factor "${INPUT_ROPE_SCALING_FACTOR}" \
    --rope_scaling_type "${INPUT_ROPE_SCALING_TYPE}" \
    --seed "${INPUT_SEED}" \
    --model_path "${INPUT_MODEL_PATH}" \
    --output_dir "${OUTPUT_OUTPUT_DIR}"

# Output "DONE"
echo "DONE"