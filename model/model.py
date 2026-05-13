import torch
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    # Load ResNet18 architecture
    model = models.resnet18(weights=None) 
    
    # Change from 2 to 4 to match your trained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4) 
    
    return model