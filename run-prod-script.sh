#!/usr/bin/env bash
set -euo pipefail

VOL_NAME="samples-volume"
HOST_DIR="./my-samples"
CONTAINER_PATH="/home/test/dataset3TSubsetSliced/sliced_dataset_dki_mppca_144_05/estimated_samples_1_ddim100"
IMAGE="andreriescoa/guided-diffusion-sample-production:sliced_dataset_dki_mppca_144_05"

# 1. Create (or confirm) the volume
docker volume create "$VOL_NAME"

# 2. Inspect to grab the host-side mountpoint
MP=$(docker volume inspect "$VOL_NAME" --format '{{ .Mountpoint }}')
echo "Volume $VOL_NAME → $MP"

# 3. Make sure the destination folder exists
mkdir -p "$HOST_DIR"

# 4. Run your container, writing into the volume
docker run --rm \
  -v "$VOL_NAME":"$CONTAINER_PATH" \
  --gpus all \
  -m 32g \
  --shm-size 2g \
  "$IMAGE"

# 5. Copy the *contents* of the volume to your host folder
sudo cp -a "${MP}/." "$HOST_DIR"
