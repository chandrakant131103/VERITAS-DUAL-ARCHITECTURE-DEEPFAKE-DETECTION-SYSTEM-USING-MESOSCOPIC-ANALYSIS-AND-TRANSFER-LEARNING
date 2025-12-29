import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Veritas | Deepfake Detector - Chandrakant",
    page_icon="🛡️",
    layout="centered"
)

# --- TITLE & HEADER ---
st.title("🛡️ Veritas: Deepfake Detection System Chandrakant")
st.markdown("### Powered by MesoNet Architecture")
st.write("Upload an image to verify if it is **Real** or **AI-Generated**.")

# --- SIDEBAR INFO ---
st.sidebar.header("About the Project")
st.sidebar.info(
    """
    This system uses a **Convolutional Neural Network (MesoNet)** trained on 
    140,000 face images to detect compression artifacts and texture 
    inconsistencies common in deepfakes.

First, the architecture and generalization are limited. You utilized MesoNet, a lightweight CNN from 2018, which is efficient but "shallow." It analyzes local texture patterns (pixel noise) rather than understanding the global context, meaning it might miss obvious anatomical errors like a person having three hands if the skin texture looks realistic. Additionally, your model is essentially a "Stable Diffusion Detector" rather than a universal AI detector. It suffers from generator bias, meaning while it excels at catching the specific models it was trained on, it may struggle to detect images from newer, unseen generators like Flux or DALL-E 3 due to poor cross-generator generalization.

Second, the system struggles with the "Smartphone Conflict," leading to false positives. Modern smartphones (like iPhones and Pixels) use aggressive computational photography techniques, such as HDR and skin smoothing, to denoise photos. To your model, this digital smoothing looks mathematically identical to the denoising artifacts found in diffusion-based deepfakes. This results in a significant flaw where high-quality, real photos are occasionally flagged as "Fake" because the model lacks the ability to distinguish between generative noise and standard mobile image processing.

Third, the model is fragile against real-world compression. MesoNet relies on detecting microscopic, high-frequency pixel artifacts to identify fakes. Social media platforms like WhatsApp or Instagram apply heavy JPEG compression that effectively "scrubs" these faint artifacts away. Consequently, while your tool works well on high-quality PNGs, its accuracy would likely drop sharply if tested on compressed images downloaded from social media, making it less robust for practical, everyday forensic use without further training on compressed data.
    """
)

# --- LOAD MODEL (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_deepfake_model():
    # Path to your saved model
    model_path = os.path.join("models", "mesonet_best.h5")
    model = load_model(model_path)
    return model

try:
    with st.spinner("Loading Model Brain..."):
        model = load_deepfake_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
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
        # Recall: < 0.5 is Fake, > 0.5 is Real (based on your folder structure)
        if prediction > 0.5:
            confidence = (prediction - 0.5) * 2 * 100
            st.success(f"✅ **RESULT: REAL FACE**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        else:
            confidence = (0.5 - prediction) * 2 * 100
            st.error(f"⚠️ **RESULT: FAKE / DEEPFAKE DETECTED**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
        # Add a progress bar for visual flair
        # Map 0-1 score to a progress bar where 0=Fake, 1=Real
        st.progress(float(prediction), text="Probability Scale (Left=Fake, Right=Real)")