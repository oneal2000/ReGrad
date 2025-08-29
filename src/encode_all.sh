#!/bin/bash

MODEL_PATH="/liuzyai04/thuir/kangjiacheng/RG/outputs/3.2-8b-1epoch"
DATA_ROOT="/liuzyai04/thuir/kangjiacheng/PRAG/data"
MODEL_NAME="8b-1epo"
TOPK=3
SPLIT="dev"
START=0
END=300

datasets=("2wikimultihopqa" "complexwebquestions" "hotpotqa" "popqa")

for dataset in "${datasets[@]}"; do
    echo "=== Running encode for $dataset ==="
    python encode.py \
        --dataset $dataset \
        --data_path $DATA_ROOT/$dataset \
        --model_path $MODEL_PATH \
        --topk $TOPK \
        --split $SPLIT \
        --start $START \
        --end $END \
        --model_name $MODEL_NAME
done
