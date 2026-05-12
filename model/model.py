import torch
from torchvision import models
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    # These four lines must be indented by exactly 4 spaces (one Tab)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# If you want to initialize it here for use within this file:
# model = Gas_leak_model()
from torchvision import models

    
    # You MUST redefine the final layer exactly as you did during training
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Assuming 2 classes: Leak/No Leak
    
    return model
