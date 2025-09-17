#!/bin/bash
set -euo pipefail

# PLM-ICD MIMIC-4 Training Script
# Usage: ./train_mimic4.sh [roberta|phi] [epochs] [optional_model_path]

MODEL_TYPE=${1:-phi}
EPOCHS=${2:-1}
MODEL_PATH_OVERRIDE=${3:-}

if [ "$EPOCHS" = "0" ]; then
  echo "Evaluating PLM-ICD ($MODEL_TYPE) with num_train_epochs=0"
else
  echo "Training PLM-ICD with MIMIC-4 data using $MODEL_TYPE model for $EPOCHS epochs"
fi

cd src

if [ "$MODEL_TYPE" = "roberta" ]; then
    DEFAULT_MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
elif [ "$MODEL_TYPE" = "phi" ]; then
    DEFAULT_MODEL_NAME="microsoft/MediPhi-MedCode"
else
    echo "Unsupported model type: $MODEL_TYPE"
    exit 1
fi

# Allow overriding the model path (e.g., to evaluate a trained checkpoint directory)
MODEL_NAME_OR_PATH=${MODEL_PATH_OVERRIDE:-$DEFAULT_MODEL_NAME}

# python3 run_icd.py \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch run_icd.py \
    --train_file ../data/mimic4/mimic4_icd10/train_full.csv \
    --validation_file ../data/mimic4/mimic4_icd10/dev_full.csv \
    --max_length 256 \
    --chunk_size 128 \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --per_device_train_batch_size 1 \
    --quantization 4bit \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs $EPOCHS \
    --num_warmup_steps 50 \
    --output_dir ../models/${MODEL_TYPE}-mimic4 \
    --model_type $MODEL_TYPE \
    --model_mode laat \
    --code_file ../data/mimic4/mimic4_icd10/ALL_CODES.txt

if [ "$EPOCHS" = "0" ]; then
  echo "Evaluation completed!"
else
  echo "Training completed! Model saved to ../models/${MODEL_TYPE}-mimic4"
fi