# Setup and train on Runpod

## Runpod overview

* You can a run a docker container in a specific region and get access to GPUs
* Each region has different kind of GPUs
* H100s seem to work well but they're the most expensive
* You have to pay for a GPU while doing prep-work not utlizing GPUs (since CPU pods can't attach network storage - see below why we need that)

## Runpod network storage

* This is to persist your data and weights even when you're not paying for a running GPU pos 
* Network storage is specific to a region
* Check what is GPU availabilty like in the region before creating a network region in it
* You'll need around 100GB
* Network storage is mounted at `/workspace`

## Create and run a pod

* Create a pod
* Make sure to attach your network storage
* Choose a GPU and count (for non-training prep work 1 is enough)
* Start pod
* SSH into the pod
* Get you code onto the pod - via e.g. `scp` or `git clone`
* Run commands (keep data in `/workspace`)
* Terminate pod to not pay when not using it
* Starting a new pod with the same network storage should keep your data (python venv and deps, training and eval data)

## Run locally (optional)

Grab the docker iamge name from your runpod template:

e.g. mine was `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

You can run the exact same docker image as a container locally:

```shell
docker run --rm -it \
-v "$(pwd):/workspace" \
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
bash
```

This way you can play around with same environment for free without paying for expensive GPU time until actually needed. You can setup scripts, etc to speed things up for real traning runs.

## Setup dependendcies (10-20mins?)

Startup pod with your network storage, SSH into it then run the following:

```shell
./setup-venv.sh # setup python venv
source .venv/bin/activate # activate venv
pip install -r requirements.txt # install requirements
```

## Get training and validation datasets (40-50 mins?)

Startup pod with your network storage, SSH into it then run the following:

```shell
source .venv/bin/activate # activate venv (unless you already have)
python hellaswag.py # get hellaswag eval dataset
python fineweb.py # get fineweb training dataset
```

## Train (? hours)

Start up pod with network storage and as many GPUs as deep your pockets are.
Set nproc_per_node to number of GPUs

```shell
source .venv/bin/activate # activate venv (unless you already have)
# train and then remove pod
torchrun --standalone --nproc_per_node=2 train_gpt2.py && runpodctl remove pod $RUNPOD_POD_ID
```
