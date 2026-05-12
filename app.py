import streamlit as st
from PIL import Image

from model.model import Gas_leak_model
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# UI SETUP
st.set_page_config(page_title="Gas Leak Detection", layout="centered")

st.title("⛽ Gas Leak Detection")
st.write("Upload an image to detect gas leaks.")

# LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# CLASS LABELS(As per the model's training)
# idx_to_class = {
#     0: "Smoke",
#     1: "Perfume",
#     2: "NoGas",
#    3: "Mixture"
# }

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning("Model not loaded.")
    else:
        tensor = preprocess(image)
        result, confidence = predict(model, tensor)
        # result, confidence = predict(model, tensor, idx_to_class)

        st.success(f"Prediction: **{result}**")
        st.write(f"Confidence: {confidence:.2%}")