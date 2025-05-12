#!/usr/bin/env bash
set -euo pipefail

DATASET_FOLDER="sliced_dataset_dki_mppca_144_05"
ESTIMATED_SAMPLES_FOLDER="estimated_samples_1_classifier_10_low_res"
HOST_DIR="dataset3TSubsetSliced/$DATASET_FOLDER/$ESTIMATED_SAMPLES_FOLDER"
IMAGE="andreriescoa/guided-diffusion-sample-classifier-production:$DATASET_FOLDER"

PORT=40088

TARGET_IP=114.32.64.6

# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080
ssh -p "$PORT" root@"$TARGET_IP" " \
docker ps -q | xargs -r docker stop \
"

ssh -p $PORT root@$TARGET_IP "mkdir -p /root/pesquisa/$HOST_DIR"

# 1. Push any files that exist locally but not on remote:
if [ -d "$HOST_DIR" ]; then
  rsync -avzP -e "ssh -p $PORT" \
    --ignore-existing \
    --append-verify \
    "$HOST_DIR"/ \
    root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/
fi

# scp -P 57709 ./$HOST_DIR root@77.104.167.149:/root/pesquisa/$HOST_DIR 

# 2. Run your container
ssh -p "$PORT" root@"$TARGET_IP" "\
  docker run --pull=always \
    -v /root/pesquisa/$HOST_DIR:/home/test/$HOST_DIR \
    --gpus all \
    -m 32g \
    --shm-size 2g \
    $IMAGE
"

# 3. Pull any files that exist on remote but not locally:
mkdir -p ./"$HOST_DIR"

rsync -avzP -e "ssh -p $PORT" \
  --ignore-existing \
  --append-verify \
  root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/ \
  ./"$HOST_DIR"/ 