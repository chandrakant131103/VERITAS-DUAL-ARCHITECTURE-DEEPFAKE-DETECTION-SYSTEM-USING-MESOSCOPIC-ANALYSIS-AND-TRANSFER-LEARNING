import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Veritas | Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- TITLE & HEADER ---
st.title("🛡️ Veritas: Deepfake Detection System")
st.markdown("### Powered by MesoNet Architecture")
st.write("Upload an image to verify if it is **Real** or **AI-Generated**.")

# --- SIDEBAR INFO ---
st.sidebar.header("About the Project")
st.sidebar.info(
    """
    This system uses a **Convolutional Neural Network (MesoNet)** trained on 
    face images to detect compression artifacts and texture 
    inconsistencies common in deepfakes.
    """
)

# --- LOAD MODEL (Robust Check) ---
@st.cache_resource
def load_deepfake_model():
    # Check if model is in 'models' folder (Local) or Root (Cloud)
    local_path = os.path.join("models", "mesonet_best.h5")
    cloud_path = "mesonet_best.h5"
    
    if os.path.exists(local_path):
        model_path = local_path
    elif os.path.exists(cloud_path):
        model_path = cloud_path
    else:
        raise FileNotFoundError("Model file 'mesonet_best.h5' not found in Root or 'models/' folder.")

    # Load the model
    model = load_model(model_path)
    return model

try:
    with st.spinner("Loading Model Brain..."):
        model = load_deepfake_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    # Stop execution if model fails
    st.stop()

# --- IMAGE PREPROCESSING ---
def prepare_image(image_data):
    # Resize to 256x256 (Same as training)
    img = ImageOps.fit(image_data, (256, 256), Image.Resampling.LANCZOS)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Show the user their image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 2. Predict Button
    if st.button("🔍 Analyze Image"):
        processed_img = prepare_image(image)
        
        # 3. Get Prediction
        prediction = model.predict(processed_img)[0][0]
        
        # 4. Display Results
        st.write("---")
        st.subheader("Forensic Analysis Report")
        
        # Determine Real vs Fake
        # THRESHOLD LOGIC: > 0.5 is REAL, < 0.5 is FAKE
        if prediction > 0.5:
            confidence = (prediction - 0.5) * 2 * 100
            st.success(f"✅ **RESULT: REAL FACE**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        else:
            confidence = (0.5 - prediction) * 2 * 100
            st.error(f"⚠️ **RESULT: FAKE / DEEPFAKE DETECTED**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
        # Add a progress bar for visual flair
        st.progress(float(prediction), text="Probability Scale (Left=Fake, Right=Real)")
