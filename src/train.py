import torch
import torch.nn as nn

from dataset import train_loader,val_loader
from model.cnn import SceneVisionCNN

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
