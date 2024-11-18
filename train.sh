#!/usr/bin/env bash

start_time=$(date +'%Y-%m-%d-%H-%M-%S')
log_file="out/${start_time}.log"
checkpoint_path=$1
max_epochs=$2

nohup bash -c \
    "(torchrun --standalone --nproc_per_node=1 train_gpt2.py $checkpoint_path $max_epochs); \
    runpodctl remove pod $RUNPOD_POD_ID" > "$log_file" 2>&1 & disown
