#!/usr/bin/env bash
set -euo pipefail

VOL_NAME="samples-volume"
HOST_DIR="dataset3TSubsetSliced/sliced_dataset_dki_mppca_144_05_b0/estimated_samples_1_ddim500"
IMAGE="andreriescoa/guided-diffusion-sample-production:sliced_dataset_dki_mppca_144_05_b0"

PORT=40088

TARGET_IP=114.32.64.6

# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080

mkdir -p ./"$HOST_DIR"

# 3. Pull any files that exist on remote but not locally:
rsync -avzP -e "ssh -p $PORT" \
  --ignore-existing \
  root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/ \
  ./"$HOST_DIR"/ 