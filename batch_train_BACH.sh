#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1
PORT=29209 bash ./tools/dist_train.sh ./swin_large_16xb64_in1k_BACH.py 1 --work-dir /home/bsj/swinTransformer_result/Second/bach/new


export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG="./swin_large_16xb64_in1k_BACH.py"
CHECKPOINT_DIR="/home/bsj/swinTransformer_result/Second/bach/new"
OUTPUT_CSV="${CHECKPOINT_DIR}/test_results.csv"
export CUDA_VISIBLE_DEVICES=1
GPUS=1

echo "checkpoint,top1_acc,precision,recall,f1-score,support,auc,confusion_matrix" > $OUTPUT_CSV

index=1

for CKPT_PATH in $(find $CHECKPOINT_DIR -name "epoch_*.pth" | sort -V); do
    echo "Testing checkpoint: ${CKPT_PATH}"

    RESULT=$(PORT=28756 bash /home/bsj/code/mmpretrain-main/tools/dist_test.sh $CONFIG $CKPT_PATH $GPUS \
         --work-dir $CHECKPOINT_DIR/test_$index)
    ((index++))
    TOP1=$(echo "$RESULT" | grep -oP 'accuracy/top1:\s*\K[0-9.]+' | head -n 1)
    PRECISION=$(echo "$RESULT" | grep -oP 'single-label/precision:\s*\K[0-9.]+' | head -n 1)
    RECALL=$(echo "$RESULT" | grep -oP 'single-label/recall:\s*\K[0-9.]+' | head -n 1)
    F1SCORE=$(echo "$RESULT" | grep -oP 'single-label/f1-score:\s*\K[0-9.]+' | head -n 1)
    SUPPORT=$(echo "$RESULT" | grep -oP 'single-label/support:\s*\K[0-9.]+' | head -n 1)
    AUC=$(echo "$RESULT" | grep -oP 'accuracy/auc:\s*\K[0-9.]+' | head -n 1)
    CONFUSION=$(echo "$RESULT" | grep -A 2 'confusion_matrix/result:' | grep -oP '[0-9]+' | paste -sd ' ')
    echo "$(basename $CKPT_PATH),${TOP1},${PRECISION},${RECALL},${F1SCORE},${SUPPORT},${AUC},${CONFUSION}" >> $OUTPUT_CSV
done























































