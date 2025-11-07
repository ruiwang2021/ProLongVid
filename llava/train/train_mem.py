from llava.train.train import train
import torch

if __name__ == "__main__":
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    train()
