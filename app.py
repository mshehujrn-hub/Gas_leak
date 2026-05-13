import streamlit as st
from PIL import Image
import torch
import os
from collections import OrderedDict
from model.model import Gas_leak_model
from utils.preprocessing import preprocess 
from huggingface_hub import hf_hub_download

# 1. Page Config
st.set_page_config(page_title="Gas Leak Detection", page_icon="⛽")
st.title("⛽ Gas Leak Detection")

@st.cache_resource
def load_model_from_hf():
    try:
        # UPDATED: Using your actual Hugging Face username and repo
        # UPDATED: Using 'Gas_leak_model.pth' with capital 'G' as seen in your upload
        model_path = hf_hub_download(
            repo_id="msquareeed/gas_leak_detection", 
            filename="Gas_leak_model.pth"
        )
        
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
        return None, str(e)
        
# Initialize global variables
model, error_message = load_model_from_hf()

# 3. UI and Prediction Logic
if error_message:
    st.error(f"❌ {error_message}")
    st.info("Ensure the file 'Gas_leak_model.pth' is fully uploaded and committed on Hugging Face.")
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

        st.write("### Prediction Probability")
        st.progress(float(probs[1]), text=f"Gas Leak: {probs[1]*100:.1f}%")
        st.progress(float(probs[0]), text=f"No Leak: {probs[0]*100:.1f}%")

        labels = ["No Leak", "Gas Leak Detected"] 
        result = labels[pred.item()]
        if result == "Gas Leak Detected":
            st.error(f"⚠️ Result: {result} ({conf*100:.2f}% confidence)")
        else:
            st.success(f"✅ Result: {result} ({conf*100:.2f}% confidence)")
