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

# 2. Load Model
model_path = os.path.join('model', 'gas_leak_model.pth')

@st.cache_resource
def load_and_configure_model():
    model = Gas_leak_model()
    if not os.path.exists(model_path):
        return None, f"File not found at {model_path}"
    try:
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

model, error_message = load_and_configure_model()

# 3. UI and Prediction Logic
if error_message:
    st.error(f"Model loading failed: {error_message}")
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
            # 'probabilities' is created HERE
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        # NOW it is safe to show the probabilities
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