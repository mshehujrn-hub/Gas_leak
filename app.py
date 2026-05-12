import torch
from collections import OrderedDict
from model.model import Gas_leak_model

# 1. Initialize the architecture skeleton
model = Gas_leak_model()

# 2. Define the path to your weights file
model_path = 'model/Gas_leak_model.pth' # Make sure this filename is exact!

try:
    # 3. Load the state_dict onto CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 4. Clean the state_dict (Removes 'module.' prefix if trained on GPU)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 5. Load weights into the skeleton
    # strict=False allows the app to run even if there are tiny key mismatches
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")