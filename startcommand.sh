#!/usr/bin/env bash
set -euo pipefail

if [[ "$RUN_MODE" == *train* ]]; then

    if [ -f "$TRAIN_DATASET_FOLDER/hr_128.tar.zst" ] && [ ! -d "$TRAIN_DATASET_FOLDER/hr_128" ]; then
        tar -I zstd -xvf "$TRAIN_DATASET_FOLDER/hr_128.tar.zst" -C "$TRAIN_DATASET_FOLDER"
    fi
    if [ -f "$TRAIN_DATASET_FOLDER/sr_16_128.tar.zst" ] && [ ! -d "$TRAIN_DATASET_FOLDER/sr_16_128" ]; then
        tar -I zstd -xvf "$TRAIN_DATASET_FOLDER/sr_16_128.tar.zst" -C "$TRAIN_DATASET_FOLDER"
    fi
    if [ -f "$TRAIN_DATASET_FOLDER/validate.tar.zst" ] && [ ! -d "$VALIDATE_DATASET_FOLDER" ]; then
        tar -I zstd -xvf "$TRAIN_DATASET_FOLDER/validate.tar.zst" -C "$TRAIN_DATASET_FOLDER"
    fi
fi


if [ "$RUN_MODE" = "train-production" ]; then
#train
    python3 scripts/super_res_train.py $TRAIN_FLAGS $SR_MODEL_FLAGS
elif [ "$RUN_MODE" = "train-debug" ]; then
    python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_train.py $TRAIN_FLAGS $SR_MODEL_FLAGS
# train classifier
elif [ "$RUN_MODE" = "train-classifier-production" ]; then
    python3 scripts/super_res_classifier_train.py $CLASSIFIER_TRAIN_FLAGS $CLASSIFIER_SR_MODEL_FLAGS
elif [ "$RUN_MODE" = "train-classifier-debug" ]; then
    python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_classifier_train.py $CLASSIFIER_TRAIN_FLAGS $CLASSIFIER_SR_MODEL_FLAGS
# sample
elif [ "$RUN_MODE" = "sample-production" ]; then \
    python3 scripts/super_res_sample.py $SAMPLE_FLAGS $SR_MODEL_FLAGS
elif [ "$RUN_MODE" = "sample-debug" ]; then \
    python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_sample.py $SAMPLE_FLAGS $SR_MODEL_FLAGS
# sample classifier
elif [ "$RUN_MODE" = "sample-classifier-production" ]; then \
    python3 scripts/super_res_classifier_sample.py $SR_MODEL_FLAGS $CLASSIFIER_SAMPLE_FLAGS $CLASSIFIER_SR_MODEL_FLAGS
elif [ "$RUN_MODE" = "sample-classifier-debug" ]; then \
    python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_classifier_sample.py $SR_MODEL_FLAGS $CLASSIFIER_SAMPLE_FLAGS $CLASSIFIER_SR_MODEL_FLAGS
else \
    echo "Unknown RUN_MODE: $RUN_MODE"; \
    exit 1; \
fi
