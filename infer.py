import sys
import torch
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def main():
    checkpoint_path = sys.argv[1]
    input_text = sys.argv[2]

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")

    from model import GPT, GPTConfig

    if (checkpoint_path == "gpt2"):
        model = GPT.from_pretrained("gpt2")
    else:
        model = GPT.from_checkpoint(checkpoint_path, device)[0]

    model.to(device)
    model.eval()

    model.generate(
        input=input_text,
        max_length=32,
        seed=42,
        num_return_sequences=4,
        enc=enc,
        device=device,
        device_type= "mps" if device == "mps" else "cuda" if device == "cuda" else "cpu",
        ddp_rank=0
    )

if __name__ == '__main__':
    main()
