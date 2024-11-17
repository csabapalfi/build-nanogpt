#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

python3 -m venv .venv

echo "Python virtual environment created at .venv."
echo
echo "Some useful commands:"
echo
echo "source .venv/bin/activate # activate venv"
echo "deactivate # deactivate venv"
echo "pip install -r requirements.txt # install requirements on venv"