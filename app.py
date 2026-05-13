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
import gdown

# 2. Robust Model Loading Logic
@st.cache_resource
def load_and_configure_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    model_path = os.path.join(model_dir, 'gas_leak_model.pth')
    
    # --- AUTO-DOWNLOAD IF MISSING ---
    if not os.path.exists(model_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Replace the ID below with your Google Drive File ID
        file_id = 'YOUR_GOOGLE_DRIVE_FILE_ID' 
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            with st.spinner("Downloading model weights (first-time setup)..."):
                gdown.download(url, model_path, quiet=False)
        except Exception as e:
            return None, f"Download failed: {e}"

    try:
        model = Gas_leak_model()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading weights: {str(e)}"
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