#!/usr/bin/env bash
set -euo pipefail

RUN_MODE=$(grep '"run_mode"' .vscode/settings.json | sed 's/.*"run_mode": "\(.*\)".*/\1/')

CONTAINER_TAG=$(grep '"container_tag"' .vscode/settings.json | sed 's/.*"container_tag": "\(.*\)".*/\1/')
echo "RUN_MODE: $RUN_MODE"
echo "CONTAINER_TAG: $CONTAINER_TAG"

IMAGE="andreriescoa/guided-diffusion-$RUN_MODE:$CONTAINER_TAG"

TMP_DIR=""

if [[ "$RUN_MODE" == *train* ]]; then
  HOST_DIR=$(sed -n 's/^ARG TRAIN_DATASET_FOLDER=\(.*\)/\1/p' Dockerfile)
  TMP_DIR="-v /root/pesquisa/tmp:/tmp"
else
  HOST_DIR=$(sed -n 's/^ARG ESTIMATED_SAMPLES_FOLDER=\(.*\)/\1/p' Dockerfile)
fi

echo "HOST_DIR: $HOST_DIR"


#PORT=43924
PORT=42841

#TARGET_IP=114.34.26.236
TARGET_IP=192.80.148.226

# ssh -p $PORT root@$TARGET_IP -L 8080:localhost:8080
ssh -p "$PORT" root@"$TARGET_IP" " \
docker ps -q | xargs -r docker stop \
"

ssh -p $PORT root@$TARGET_IP "mkdir -p /root/pesquisa/$HOST_DIR"

# 1. Push any files that exist locally but not on remote:
if [[ "$RUN_MODE" == *sample* ]] && [ -d "$HOST_DIR" ]; then
  rsync -avzP -e "ssh -p $PORT" \
    --ignore-existing \
    --append-verify \
    "$HOST_DIR"/ \
    root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/
fi


# scp -P 57709 ./$HOST_DIR root@77.104.167.149:/root/pesquisa/$HOST_DIR 

# 2. Run your container
ssh -t -p "$PORT" root@"$TARGET_IP" "\
  docker run --pull=always \
    -v /root/pesquisa/$HOST_DIR:/home/test/$HOST_DIR \
    $TMP_DIR \
    --gpus all \
    -m 32g \
    --shm-size 2g \
    $IMAGE
"

# 3. Pull any files that exist on remote but not locally:
if [[ "$RUN_MODE" == *train* ]]
  HOST_DIR=$HOST_DIR/tmp
fi

mkdir -p ./"$HOST_DIR"

rsync -avzP -e "ssh -p $PORT" \
  --ignore-existing \
  --append-verify \
  root@$TARGET_IP:/root/pesquisa/"$HOST_DIR"/ \
  ./"$HOST_DIR"/ 