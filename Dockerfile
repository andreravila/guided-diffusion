##
# Directory of the application inside container
##
ARG APP_ROOT=/home/test

ARG RUN_MODE=production

ARG DATASET_FOLDER=

ARG ESTIMATED_SAMPLES_FOLDER=

#FROM pytorch/pytorch

FROM ubuntu:22.04

ARG APP_ROOT

# Install required packages
RUN apt-get -q -y update && \
    apt-get -q -y install \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libopenmpi-dev \
    && apt-get clean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_ROOT}

# Create virtualenv
RUN python3 -m venv .venv

# COPY and install requirements
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN . .venv/bin/activate && \
    pip3 install --upgrade pip && pip install -e . && \
    pip3 install -r requirements.txt 

ARG DATASET_FOLDER
# copy the checkpoints
COPY checkpoint_model/${DATASET_FOLDER}/model checkpoint_model/${DATASET_FOLDER}/model

# copy the test part of the dataset, to run the container directly
COPY dataset3TSubsetSliced/${DATASET_FOLDER}/test_1 dataset3TSubsetSliced/${DATASET_FOLDER}/test_1

# copy the rest of the application
COPY scripts scripts
COPY guided_diffusion guided_diffusion


ARG RUN_MODE

ENV TRAIN_FLAGS="--lr_anneal_steps 100000 --batch_size 128 --lr 1e-5 --save_interval 5000 --weight_decay 0.05"

# ENV SAMPLE_FLAGS="--batch_size 4"
# using ddim
ENV SAMPLE_FLAGS="--batch_size 12 --timestep_respacing ddim500 --use_ddim True"

ENV CLASSIFIER_TRAIN_FLAGS="--iterations 100000 --anneal_lr True --batch_size 128 --lr 1e-5 --save_interval 5000 --weight_decay 0.05"

ENV CLASSIFIER_SAMPLE_FLAGS="--batch_size 4"

# Acording to what was tested in the paper, can also be, instead of --num_channels 192, --num_channels 256
ENV SR_MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 2000 --large_size 128 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

ENV CLASSIFIER_SR_MODEL_FLAGS="--large_size 128 --small_size 128 --diffusion_steps 2000 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True"

# change the RUN_MODE at .vscode/settings.json
RUN if [ "$RUN_MODE" = "train-production" ]; then \
        echo "python3 scripts/super_res_train.py $TRAIN_FLAGS $SR_MODEL_FLAGS" > startcommand.sh; \
    elif [ "$RUN_MODE" = "train-debug" ]; then \
    echo "python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_train.py $TRAIN_FLAGS $SR_MODEL_FLAGS" > startcommand.sh; \
    # classifier
    elif [ "$RUN_MODE" = "train-classifier-production" ]; then \
        echo "python3 scripts/super_res_classifier_train.py $CLASSIFIER_TRAIN_FLAGS $CLASSIFIER_SR_MODEL_FLAGS" > startcommand.sh; \
    elif [ "$RUN_MODE" = "train-classifier-debug" ]; then \
        echo "python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_classifier_train.py $CLASSIFIER_TRAIN_FLAGS $CLASSIFIER_SR_MODEL_FLAGS" > startcommand.sh; \
    # sample
    elif [ "$RUN_MODE" = "sample-production" ]; then \
        echo "python3 scripts/super_res_sample.py $SAMPLE_FLAGS $SR_MODEL_FLAGS" > startcommand.sh; \
    elif [ "$RUN_MODE" = "sample-debug" ]; then \
        echo "python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_sample.py $SAMPLE_FLAGS $SR_MODEL_FLAGS" > startcommand.sh; \
    # sample classifier
    elif [ "$RUN_MODE" = "sample-classifier-production" ]; then \
        echo "python3 scripts/super_res_classifier_sample.py $SR_MODEL_FLAGS $CLASSIFIER_SAMPLE_FLAGS $CLASSIFIER_SR_MODEL_FLAGS" > startcommand.sh; \
    elif [ "$RUN_MODE" = "sample-classifier-debug" ]; then \
        echo "python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_classifier_sample.py $SR_MODEL_FLAGS $CLASSIFIER_SAMPLE_FLAGS $CLASSIFIER_SR_MODEL_FLAGS" > startcommand.sh; \
    else \
        echo "Unknown RUN_MODE: $RUN_MODE"; \
        exit 1; \
    fi

RUN chmod +x startcommand.sh

# Activate and run the code
CMD . .venv/bin/activate && ./startcommand.sh

# command to execute the container
# docker build --build-arg RUN_MODE=production -t guided-diffusion .
#  docker run -v ./checkpoint_model:/home/test/checkpoint_model -v ./tmp:/tmp -v ./dataset3TSubsetSliced:/home/test/dataset3TSubsetSliced --gpus all -m=32g  --shm-size=2g guided-diffusion-production
# consult vscode tasks.json