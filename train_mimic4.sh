#!/bin/bash

# PLM-ICD MIMIC-4 Training Script
# Usage: ./train_mimic4.sh [roberta|phi] [epochs]

MODEL_TYPE=${1:-phi}
EPOCHS=${2:-1}

echo "Training PLM-ICD with MIMIC-4 data using $MODEL_TYPE model for $EPOCHS epochs"

cd src

if [ "$MODEL_TYPE" = "roberta" ]; then
    MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
elif [ "$MODEL_TYPE" = "phi" ]; then
    MODEL_NAME="microsoft/MediPhi-MedCode"
else
    echo "Unsupported model type: $MODEL_TYPE"
    exit 1
fi

# python3 run_icd.py \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch run_icd.py \
    --train_file ../data/mimic4/mimic4_icd10/train_full.csv \
    --validation_file ../data/mimic4/mimic4_icd10/dev_full.csv \
    --max_length 256 \
    --chunk_size 128 \
    --model_name_or_path $MODEL_NAME \
    --per_device_train_batch_size 1 \
    --quantization 8bit \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --fp16 True \
    --per_device_eval_batch_size 1 \
    --num_train_epochs $EPOCHS \
    --num_warmup_steps 50 \
    --output_dir ../models/${MODEL_TYPE}-mimic4 \
    --model_type $MODEL_TYPE \
    --model_mode laat \
    --code_file ../data/mimic4/mimic4_icd10/ALL_CODES.txt

echo "Training completed! Model saved to ../models/${MODEL_TYPE}-mimic4"