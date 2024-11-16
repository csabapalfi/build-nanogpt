#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Check if the current working directory is /workspace
if [ "$(pwd)" != "/workspace" ]; then
    echo "This script must be run from /workspace."
    exit 1
fi

python3 -m venv .venv
echo "Python virtual environment created at .venv."
echo
echo "Some useful commands:"
echo
echo "source .venv/bin/activate # activate venv"
echo "deactivate # deactivate venv"
echo "pip install -r requirements.txt # install requirements"