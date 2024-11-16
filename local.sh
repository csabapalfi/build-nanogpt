#!/usr/bin/env bash

docker run --rm -it \
-v "$(pwd):/workspace" \
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
bash