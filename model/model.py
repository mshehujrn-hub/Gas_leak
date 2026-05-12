import torch
import torch.nn as nn
from torchvision import models

def Gas_leak_model():
    # 1. Initialize the base model
    model = models.resnet18(weights=None) 
    
    # 2. Define the input features for the final layer
    num_ftrs = model.fc.in_features
    
    # 3. Define your classes (must be defined before use!)
    class_names = ['Leak', 'No Leak'] 
    
    # 4. Rebuild the final layer
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    return model
