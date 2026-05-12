import streamlit as st
import torch
import os
from collections import OrderedDict
from model.model import Gas_leak_model

# 1. Initialize Streamlit page config (Always first)
st.set_page_config(page_title="Gas Leak Detection")
st.title("⛽ Gas Leak Detection")

# 2. Define the exact path found in your logs
# We use lowercase 'gas_leak_model.pth' to match your folder list
model_path = os.path.join('model', 'gas_leak_model.pth')

@st.cache_resource
def load_and_configure_model():
    # Initialize the architecture
    model = Gas_leak_model()
    
    if not os.path.exists(model_path):
        return None, f"File not found at {model_path}"

    try:
        # Load weights for CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Clean module prefix if necessary
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# 3. Execution logic
model, error_message = load_and_configure_model()

if error_message:
    st.error(f"Model loading failed: {error_message}")
    st.info("Check if your file name is exactly 'gas_leak_model.pth' in lowercase.")
elif model:
    st.success("Model loaded successfully!")
    # Proceed with your file_uploader and prediction logic here...
    # --- IMAGE UPLOADER SECTION ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")

    # 2. Preprocess the image
    # Note: Ensure you have your 'preprocess' function imported from utils
    from utils.preprocessing import preprocess 
    input_tensor = preprocess(image)

    # 3. Perform Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # 4. Map the result to a label
    # Adjust these labels to match your specific training classes
    labels = ["No Leak", "Gas Leak Detected"] 
    result = labels[predicted_class.item()]

    # 5. Display the Results
    if result == "Gas Leak Detected":
        st.error(f"⚠️ Result: {result} ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"✅ Result: {result} ({confidence*100:.2f}% confidence)")