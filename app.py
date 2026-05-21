import streamlit as st
from PIL import Image
import torch
import os
from collections import OrderedDict
from model.model import Gas_leak_model
from utils.preprocessing import preprocess 
from huggingface_hub import hf_hub_download

# 1. Page Config
st.set_page_config(page_title="Gas Leak Detector", page_icon="⛽")
st.title("⛽ Gas Leak Detector")

@st.cache_resource
def load_model_from_hf():
    try:
        # Downloads from your Hugging Face repo
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

        # 4. UPDATED: Display 4 Probabilities
        st.write("### Prediction Probabilities")
        
        # Mapping labels to your 4 classes
        # NOTE: Update these names to match your training folder names
        labels = ["No Leak", "Minor Leak", "Critical Leak", "Other/Noise"] 
        
        for i in range(len(labels)):
            st.progress(float(probs[i]), text=f"{labels[i]}: {probs[i]*100:.1f}%")

        # 5. Final Result
        result = labels[pred.item()]
        if "Leak" in result and "No Leak" not in result:
            st.error(f"⚠️ Result: {result} ({conf*100:.2f}% confidence)")
        else:
            st.success(f"✅ Result: {result} ({conf*100:.2f}% confidence)")