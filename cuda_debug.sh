#!/usr/bin/env bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
