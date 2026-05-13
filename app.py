import streamlit as st
from PIL import Image
import torch
import os
from collections import OrderedDict
from model.model import Gas_leak_model
from utils.preprocessing import preprocess 

# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="Gas Leak Detection")
st.title("⛽ Gas Leak Detection")

# --- PATH FIX START ---
# This ensures the app finds the model folder regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'gas_leak_model.pth')
# --- PATH FIX END ---

@st.cache_resource
def load_and_configure_model():
    # Initialize the architecture
    model = Gas_leak_model()
    
    # Check if file exists before trying to load
    if not os.path.exists(model_path):
        return None, f"File not found at {model_path}. Please check your GitHub repository structure."
    
    try:
        # Load weights
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Clean state dict (removes 'module.' prefix if trained with DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

model, error_message = load_and_configure_model()

# 3. UI and Prediction Logic
if error_message:
    st.error(f"Model loading failed: {error_message}")
    # Debugging info for you to see in the UI
    st.info(f"Checking directory: {BASE_DIR}")
elif model:
    st.success("Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Perform Inference
        input_tensor = preprocess(image)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        # UI Output
        st.write("### Prediction Probability")
        st.progress(float(probabilities[1]), text=f"Gas Leak: {probabilities[1]*100:.1f}%")
        st.progress(float(probabilities[0]), text=f"No Leak: {probabilities[0]*100:.1f}%")

        # Display Final Result
        labels = ["No Leak", "Gas Leak Detected"] 
        result = labels[predicted_class.item()]
        if result == "Gas Leak Detected":
            st.error(f"⚠️ Result: {result} ({confidence*100:.2f}% confidence)")
        else:
            st.success(f"✅ Result: {result} ({confidence*100:.2f}% confidence)")