#!/usr/bin/env bash
set -euo pipefail

DATASET_FOLDER="sliced_dataset_dki_mppca_144_05"
ESTIMATED_SAMPLES_FOLDER="estimated_samples_1_classifier_10_low_res"
HOST_DIR="dataset3TSubsetSliced/$DATASET_FOLDER/$ESTIMATED_SAMPLES_FOLDER"
IMAGE="andreriescoa/guided-diffusion-sample-classifier-production:$DATASET_FOLDER"

PORT=40088

TARGET_IP=114.32.64.6

# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080

mkdir -p ./"$HOST_DIR"

# 3. Pull any files that exist on remote but not locally:
rsync -avzP -e "ssh -p $PORT" \
  --ignore-existing \
  --append-verify \
  root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/ \
  ./"$HOST_DIR"/ 