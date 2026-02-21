import torch

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"

print(DEVICE)