import torch
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    # If you trained on ResNet18, use resnet18 here
    model = models.resnet18(weights=None) 
    
    # You MUST redefine the final layer exactly as you did during training
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Assuming 2 classes: Leak/No Leak
    
    return model
