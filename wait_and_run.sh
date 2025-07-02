#!/bin/bash

if [ -z "$1" ]; then
    echo "[ERROR] Please provide a numeric experiment ID as the first argument."
    echo "Usage: ./wait_and_ren.sh 5"
    exit 1
fi

EXP_ID="$1"
LOG_FILE="logs/matchsum_exp_${EXP_ID}.log"

COMMAND="CUDA_LAUNCH_BLOCKING=1 nohup python train.py --config-name train_config > ${LOG_FILE}"

echo "[INFO] Waiting for all GPUs to be idle..."

while true; do
    NUM_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')

    if [ "$NUM_PROCS" -eq 0 ]; then
        echo "[INFO] All GPUs are idle. Starting the job..."
        break
    else
        echo "[INFO] GPUs are still in use... checking again in 10 minutes."
        sleep 600
    fi
done

echo "[INFO] Launching MatchSum training..."
eval $COMMAND