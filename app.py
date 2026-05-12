import os
import streamlit as st
import torch
from model.model import Gas_leak_model

# 1. Debug: Let's see what files are actually in the model folder
if os.path.exists('model'):
    st.write("Files found in model folder:", os.listdir('model'))

model_path = 'model/Gas_leak_model.pth' 

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_gas_model():
    model = Gas_leak_model()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error(f"File not found at {model_path}. Please check your GitHub folder.")
        return None

model = load_gas_model()

if model:
    st.success("Gas Leak Detection Model is ready!")