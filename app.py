import streamlit as st
from PIL import Image
import torch
import os
import gdown
from collections import OrderedDict
from model.model import Gas_leak_model
from utils.preprocessing import preprocess 

# 1. Page Config
st.set_page_config(page_title="Gas Leak Detection", page_icon="⛽")
st.title("⛽ Gas Leak Detection")

# 2. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'gas_leak_model.pth')

@st.cache_resource
def get_model():
    # Ensure folder exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # This is the ID for your gas_leak_model.pth file
            
    # --- DOWNLOAD FROM GOOGLE DRIVE ---
    # Replace 'YOUR_FILE_ID' with the actual ID from your Google Drive link
    if not os.path.exists(MODEL_PATH):
        file_id = '1eT-y9-O-S-wG3S9v_I-V6K8J3k0G7y8Z' 
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner("Downloading model weights from Google Drive..."):
                gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            return None, f"Download failed: {str(e)}"

    # --- LOAD PRE-TRAINED MODEL ---
    try:
        model = Gas_leak_model()
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Strip 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Model load error: {str(e)}"

# Define the variables globally
model, error_message = get_model()

# 3. UI and Prediction Logic
if error_message:
    st.error(f"❌ {error_message}")
    st.info("Check your Google Drive ID and sharing permissions (Anyone with link).")
elif model:
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with st.spinner('Analyzing...'):
            input_tensor = preprocess(image)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(probs, 0)

        # UI Results
        st.write("### Prediction Probability")
        st.progress(float(probs[1]), text=f"Gas Leak: {probs[1]*100:.1f}%")
        st.progress(float(probs[0]), text=f"No Leak: {probs[0]*100:.1f}%")

        labels = ["No Leak", "Gas Leak Detected"] 
        result = labels[pred.item()]
        if result == "Gas Leak Detected":
            st.error(f"⚠️ Result: {result} ({conf*100:.2f}% confidence)")
        else:
            st.success(f"✅ Result: {result} ({conf*100:.2f}% confidence)")