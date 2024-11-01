##
# Directory of the application inside container
##
ARG APP_ROOT=/home/test

ARG RUN_MODE=production

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

# copy the rest of the application
COPY scripts scripts
COPY guided_diffusion guided_diffusion

# copy the dataset
# COPY dataset3TSubsetSliced dataset3TSubsetSliced

#ENV startcommand='python3 scripts/super_res_train.py --large_size 128 --small_size 128 --diffusion_steps 1000'

#ENV startcommand='python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_train.py --large_size 128 --small_size 128 --diffusion_steps 1000'
 
ARG RUN_MODE

RUN if [ "$RUN_MODE" = "production" ]; then \
        echo "python3 scripts/super_res_train.py --large_size 128 --small_size 128 --diffusion_steps 1000" > startcommand.sh; \
    elif [ "$RUN_MODE" = "debug" ]; then \
        echo "python3 -m debugpy --listen 0.0.0.0:6502 --log-to src/log --wait-for-client scripts/super_res_train.py --large_size 128 --small_size 128 --diffusion_steps 1000" > startcommand.sh; \
    else \
        echo "Unknown RUN_MODE: $RUN_MODE"; \
        exit 1; \
    fi

RUN chmod +x startcommand.sh

# Activate and run the code
CMD . .venv/bin/activate && ./startcommand.sh

# command to execute the container
# docker build --build-arg RUN_MODE=production -t guided-diffusion .
# docker run -v ./tmp:/tmp -v ./dataset3TSubsetSliced:/home/test/dataset3TSubsetSliced --gpus all guided-diffusion
# consult vscode tasks.json

# TODO: Move the debug to here also