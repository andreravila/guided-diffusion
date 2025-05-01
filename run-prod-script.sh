#!/usr/bin/env bash
set -euo pipefail

VOL_NAME="samples-volume"
HOST_DIR="dataset3TSubsetSliced/sliced_dataset_dki_mppca_144_05_b0/estimated_samples_1_ddim500"
IMAGE="andreriescoa/guided-diffusion-sample-production:sliced_dataset_dki_mppca_144_05_b0"

PORT=40088

TARGET_IP=114.32.64.6

# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080
ssh -p "$PORT" root@"$TARGET_IP" " \
docker stop $(docker ps -a -q) \
"

ssh -p $PORT root@$TARGET_IP "mkdir -p /root/pesquisa/$HOST_DIR"

# 1. Push any files that exist locally but not on remote:
if [ -d "$HOST_DIR" ]; then
  rsync -avzP -e "ssh -p $PORT" \
    --ignore-existing \
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
  root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/ \
  ./"$HOST_DIR"/ 