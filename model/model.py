import torch
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    # Must match training: resnet18, resnet34, or resnet50
    model = models.resnet18(weights=None) 
    
    # Must match the number of classes you trained with
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    return model