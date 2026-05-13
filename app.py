import streamlit as st
from PIL import Image
import torch
import os
from collections import OrderedDict
from model.model import Gas_leak_model
from utils.preprocessing import preprocess 

# 1. Page Config (MUST be first)
st.set_page_config(page_title="Gas Leak Detection", page_icon="⛽")
st.title("⛽ Gas Leak Detection")

# 2. Robust Model Loading Logic
@st.cache_resource
def load_and_configure_model():
    # Define every possible place the model could be on the Streamlit server
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, 'model', 'gas_leak_model.pth'),
        'model/gas_leak_model.pth',
        '/mount/src/gas_leak/model/gas_leak_model.pth'
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        # Diagnostic check: see what the server actually sees
        folder_path = os.path.join(base_dir, 'model')
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            return None, f"Folder exists but file not found. Found in /model: {files}"
        return None, f"Model folder not found at {folder_path}. Check GitHub casing (model vs Model)."

    try:
        # Initialize model architecture
        model = Gas_leak_model()
        
        # Load weights to CPU
        state_dict = torch.load(actual_path, map_location=torch.device('cpu'))
        
        # Strip 'module.' prefix from DataParallel training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading weights: {str(e)}"

# Execute the loader
model, error_message = load_and_configure_model()

# 3. UI and Prediction Logic
if error_message:
    st.error(f"❌ {error_message}")
    st.info("💡 Tip: Ensure your model file is pushed to GitHub and is not just a Git LFS pointer.")
elif model:
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Perform Inference
        with st.spinner('Analyzing image...'):
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