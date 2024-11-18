#!/usr/bin/env bash

start_time=$(date +'%Y-%m-%d-%H-%M-%S')
log_file="out/${start_time}.log"

nohup bash -c \
    "(torchrun --standalone --nproc_per_node=2 train_gpt2.py); \
    runpodctl remove pod $RUNPOD_POD_ID" > "$log_file" 2>&1 & disown
