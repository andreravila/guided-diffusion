##
# Directory of the application inside container
##
ARG APP_ROOT=/home/test

#FROM pytorch/pytorch

FROM ubuntu:22.04

ARG IMAGE_ARCH
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

# copy the rest of the application
COPY scripts scripts
COPY guided_diffusion guided_diffusion

# copy the dataset
# COPY dataset3TSubsetSliced dataset3TSubsetSliced

# Activate and run the code
CMD . .venv/bin/activate && python3 scripts/super_res_train.py --large_size 128 --small_size 128 --diffusion_steps 2000

# command to execute the container
# docker run -v ./tmp:/tmp -v ./dataset3TSubsetSliced:/home/test/dataset3TSubsetSliced --gpus all guided-diffusion

# TODO: Move the debug to here also