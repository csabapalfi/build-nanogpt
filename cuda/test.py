import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)  # Shows the CUDA version PyTorch was built with
print("Available CUDA devices:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA device available")
print("CUDA is available:", torch.cuda.is_available())  # Check if CUDA is available
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")