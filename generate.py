import sys
import torch
import tiktoken
from model import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")

def main():
    checkpoint_path = sys.argv[1]
    input_text = sys.argv[2]

    if (checkpoint_path == "gpt2"):
        model = GPT.from_pretrained("gpt2")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = GPT(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    device_type = "mps" if device == "mps" else "cuda" if device == "cuda" else "cpu"
    model.to(device)
    model.eval()

    model.generate(
        input=input_text,
        max_length=32,
        seed=42,
        num_return_sequences=4,
        enc=enc,
        device=device,
        device_type=device_type,
        ddp_rank=0
    )

if __name__ == '__main__':
    main()
