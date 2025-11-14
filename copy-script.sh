#!/usr/bin/env bash
set -euo pipefail

RUN_MODE=$(grep '"run_mode"' .vscode/settings.json | sed 's/.*"run_mode": "\(.*\)".*/\1/')

CONTAINER_TAG=$(grep '"container_tag"' .vscode/settings.json | sed 's/.*"container_tag": "\(.*\)".*/\1/')
echo "RUN_MODE: $RUN_MODE"
echo "CONTAINER_TAG: $CONTAINER_TAG"

IMAGE="andreriescoa/guided-diffusion-$RUN_MODE:$CONTAINER_TAG"

SRC_DIR=""

DATASET_FOLDER=$(sed -n 's/^ARG DATASET_FOLDER=\(.*\)/\1/p' Dockerfile)

if [[ "$RUN_MODE" == *train-classifier* ]]; then
  HOST_DIR=$(sed -n 's/^ARG CLASSIFIER_DIR=\(.*\)/\1/p' Dockerfile)
  # Replace literal "${DATASET_FOLDER}" in HOST_DIR with the actual variable value
  HOST_DIR="${HOST_DIR//\$\{DATASET_FOLDER\}/$DATASET_FOLDER}"
  SRC_DIR="tmp-copy-classifier"
elif [[ "$RUN_MODE" == *train* ]]; then
  HOST_DIR=$(sed -n 's/^ARG MODEL_DIR=\(.*\)/\1/p' Dockerfile)
  # Replace literal "${DATASET_FOLDER}" in HOST_DIR with the actual variable value
  HOST_DIR="${HOST_DIR//\$\{DATASET_FOLDER\}/$DATASET_FOLDER}"
  SRC_DIR="tmp-copy"
else
  HOST_DIR=$(sed -n 's/^ARG ESTIMATED_SAMPLES_FOLDER=\(.*\)/\1/p' Dockerfile)
  # Replace literal "${DATASET_FOLDER}" in HOST_DIR with the actual variable value
  HOST_DIR="${HOST_DIR//\$\{DATASET_FOLDER\}/$DATASET_FOLDER}"
  SRC_DIR=$HOST_DIR
fi

echo "HOST_DIR: $HOST_DIR"
echo "SRC_DIR: $SRC_DIR"


# train 01 b0 half
PORT=40738
TARGET_IP=84.2.197.162
GPU_DEVICE="device=0"




# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080

mkdir -p ./"$HOST_DIR"

# 3. Pull any files that exist on remote but not locally:
rsync -avzP -e "ssh -p $PORT" \
  --ignore-existing \
  --append-verify \
  root@$TARGET_IP:/root/pesquisa/"$SRC_DIR"/ \
  ./"$HOST_DIR"/ 