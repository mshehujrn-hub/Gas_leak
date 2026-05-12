import torch.nn as nn
import torch.optim as optim
from torchvision import models # Added missing import for models

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features

# Changing the output layer to 4, based on the number of class_names
model.fc = nn.Linear(num_ftrs, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)