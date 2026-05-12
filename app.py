import os
import streamlit as st
import torch
from model.model import Gas_leak_model

# This will print the contents of your model folder directly on the app screen
if os.path.exists('model'):
    files = os.listdir('model')
    st.write(f"Files found in /model folder: {files}")
else:
    st.error("The folder 'model' does not exist on GitHub!")

# Change this line to be all lowercase
model_filename = 'gas_leak_model.pth' 
model_path = os.path.join('model', model_filename)

@st.cache_resource
def load_my_model():
    if os.path.exists(model_path):
        model = Gas_leak_model()
        # map_location='cpu' is mandatory for Streamlit Cloud
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    return None

model = load_my_model()
if model:
    st.success("Target acquired: Model loaded!")

    