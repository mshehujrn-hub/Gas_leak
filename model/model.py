import torch
from model.model import Gas_leak_model

# 1. Initialize the architecture
model = Gas_leak_model()

# 2. Load the weights with map_location
try:
    state_dict = torch.load('model/your_weights_file.pth', map_location=torch.device('cpu'))
    
    # If you see "module." in your error keys, use this fix:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
