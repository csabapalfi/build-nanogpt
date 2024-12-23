# Field Notes

## <a name='tldr'></a>tl;dr

* trained a GPT2 model for 2 epochs on the fineweb dataset
* used runpod for about ~$40 per epoch (2x H100 GPUs and 100GB network storage)

### <a name='results'></a>results

* see [notebooks/plots.ipynb](notebooks/plots.ipynb) for loss, norm and learning rate plots
* achieved a validation loss of 3.0167 and a hella score of 0.3172
* shame I forgot to increase the learning rate (like Andrej mentioned)

### <a name='lessonslearnt'></a>lessons learnt

* VSCode Remote SSH worked like a charm
* local setup is free and can work for non-GPU prep work and limited debugging
* tried and the model can be sampled on Apple Silicon Macs via MPS
* used `runpodctl remove pod` to self-destruct the pod after overnight training to save money
* cuda/torch version mismatches could easily trip you up, careful with those

## <a name='contents'></a>contents

<!-- vscode-markdown-toc -->
- [Field Notes](#field-notes)
	- [tl;dr](#tldr)
		- [results](#results)
		- [lessons learnt](#lessons-learnt)
	- [contents](#contents)
	- [runpod](#runpod)
		- [runpod network storage](#runpod-network-storage)
		- [create and run a pod](#create-and-run-a-pod)
	- [train your model](#train-your-model)
		- [setup dependendcies (10-20mins)](#setup-dependendcies-10-20mins)
		- [get training and validation datasets (40-50 mins)](#get-training-and-validation-datasets-40-50-mins)
		- [train epoch 1 (4 hours with 2 H100s)](#train-epoch-1-4-hours-with-2-h100s)
		- [train epoch 2 (4 hours with 2 H100s)](#train-epoch-2-4-hours-with-2-h100s)
		- [plot loss, norm and learning rate](#plot-loss-norm-and-learning-rate)
	- [neat tricks](#neat-tricks)
		- [local prep with a runpod docker image](#local-prep-with-a-runpod-docker-image)
		- [autokill your pod when training is done](#autokill-your-pod-when-training-is-done)
		- [local sampling from the model](#local-sampling-from-the-model)
		- [use vscode via remote ssh](#use-vscode-via-remote-ssh)
		- [debug cuda errors](#debug-cuda-errors)
	- [TODO](#todo)
		- [smarter epoch handling and resume training](#smarter-epoch-handling-and-resume-training)
		- [warning - loading checkpoints with weights\_only=False](#warning---loading-checkpoints-with-weights_onlyfalse)
		- [MPS autocast warning on local Mac](#mps-autocast-warning-on-local-mac)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


## <a name='runpod'></a>runpod

* See https://www.runpod.io/
* You can a run a docker container in a specific region and get access to GPUs
* Each region has different kind of GPUs
* H100s seem to work well but they're the most expensive

### <a name='runpodnetworkstorage'></a>runpod network storage

* This is to persist your data and weights (even when you're not paying for a running GPU pod)
* The idea is to start a pod with the storage attached, do some work, terminate pod and you can start another pod with the same data later

* Network storage is specific to a region
* Check what is GPU availabilty like in the region before creating a network region in it
* I configured 100GB to be sure and ended up with this much used:
    * python packages and venv - 8GB
    * fineweb about 20GB after setup completed (+ additional 30GB while setting up)
    * model size 500MB (times as many checkpoints you have)
* Network storage is mounted at `/workspace`

### <a name='createandrunapod'></a>create and run a pod

* Create a pod
* Make sure to attach your network storage
* Choose a GPU and count (for non-training prep work 1 is enough [1])
* Start pod and SSH into it
* Get you code onto the pod - via e.g. `scp` or `git clone` OR use vscode remote ssh
* Run commands (keep data in `/workspace`)
* Terminate pod to not pay when not using it
* Starting a new pod with the same network storage should keep your data (python venv and deps, training and eval data)

[1] for some (most) prep work you don't even need a GPU but CPU pods can't attach to network storage at the moment, you can choose a single GPU for this to minimize costs

## <a name='trainyourmodel'></a>train your model

### <a name='setupdependendcies10-20mins'></a>setup dependendcies (10-20mins)

Startup pod with your network storage, SSH into it then run the following:

```shell
python -m venv .venv # setup python venv
source .venv/bin/activate # activate venv
pip install -r requirements.txt # install requirements
```

### <a name='gettrainingandvalidationdatasets40-50mins'></a>get training and validation datasets (40-50 mins)

Startup pod with your network storage, SSH into it then run the following:

```shell
source .venv/bin/activate # activate venv (unless you already have)
python data/hellaswag.py # get hellaswag eval dataset
python data/fineweb.py # get fineweb training dataset
```

### <a name='trainepoch14hourswith2H100s'></a>train epoch 1 (4 hours with 2 H100s)

Start up pod with network storage and as many GPUs as deep your pockets are.
Set nproc_per_node to number of GPUs

```shell
source .venv/bin/activate # activate venv (unless you already have)
./train.sh
exit # you can logout and your training will continue in the background then remove pod after training to save money
```

### <a name='trainepoch24hourswith2H100s'></a>train epoch 2 (4 hours with 2 H100s)

```shell
./train.sh model_19072.pt 2
```

### <a name='plotlossnormandlearningrate'></a>plot loss, norm and learning rate

* see [notebooks/plots.ipynb](notebooks/plots.ipynb) for loss, norm and learning rate plots

## <a name='neattricks'></a>neat tricks

### <a name='localprepwitharunpoddockerimage'></a>local prep with a runpod docker image

Grab the docker image name from your runpod template:

e.g. mine was `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

You can run the exact same docker image as a container locally:

```shell
docker run --rm -it \
-v "$(pwd):/workspace" \
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
bash
```

This way you can play around with same environment for free without paying for expensive GPU time until actually needed. You can setup scripts, etc to speed things up for real traning runs.

I've used this to create some of my scripts and test a few things.

### <a name='autokillyourpodwhentrainingisdone'></a>autokill your pod when training is done

See `train.sh` on how to do this.

Basically each runpod container comes with `runpodctl` command and the `RUNPOD_POD_ID` env var available.

The command `runpodctl remove pod $RUNPOD_POD_ID` automates removing your pod e.g. at the end of our training to save money.

### <a name='localsamplingfromthemodel'></a>local sampling from the model

* download the checkpoint from your runpod GPU pod (e.g via scp)
* run `python infer.py <checkpoint_path> <input_text>`
* you can even compare with real gpt2: `python infer.py gpt2 <input_text>`

This runs pretty fast on a M3 Mac given how 'small' our model is:

```shell
time python infer.py checkpoints/model_19072.pt "Hello, I'm a language model,"
Using device: mps
# skipping output (some warnings and the actual samples)...

real	0m9.055s
user	0m6.655s
sys	0m1.064s
```

### <a name='usevscodeviaremotessh'></a>use vscode via remote ssh

* just like Andrej
* Install [Remote - SSH from VS Marketplace](https://marketplace.cursorapi.com/items?itemName=ms-vscode-remote.remote-ssh)
* Open vscode, go to `Remote-SSH: Connect to Host...` and add the ssh command for your runpod
* Open folder `/workspace`

### <a name='debugcudaerrors'></a>debug cuda errors

* I've run into some when trying a different GPU
* check you versions via some scripts I added `cuda/test.sh` and `cuda/test.py`

## <a name='TODO'></a>TODO

### <a name='smarterepochhandlingandresumetraining'></a>smarter epoch handling and resume training

* currently it's a bit hacky to resume training and it only supports full epochs
* learning rate scheduling was not adjusted to epochs
* checkpointing should also save optimizer.state_dict() and rng seeds etc. if you want to more exactly resume training
* permutate training data more at the start of each epoch
* increased learning rate mentioned by Andrej

### <a name='warning-loadingcheckpointswithweights_onlyFalse'></a>warning - loading checkpoints with weights_only=False

```
/Users/csabapalfi/yc/build-nanogpt/generate.py:15: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
```

### <a name='MPSautocastwarningonlocalMac'></a>MPS autocast warning on local Mac

We already autocast to bfloat16 but we still get this warning.

```
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
/opt/homebrew/lib/python3.10/site-packages/torch/amp/autocast_mode.py:332: UserWarning: In MPS autocast, but the target dtype is not supported. Disabling autocast.
MPS Autocast only supports dtype of torch.bfloat16 currently.
  warnings.warn(error_message)
```