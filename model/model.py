import torch
from torchvision import models
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model