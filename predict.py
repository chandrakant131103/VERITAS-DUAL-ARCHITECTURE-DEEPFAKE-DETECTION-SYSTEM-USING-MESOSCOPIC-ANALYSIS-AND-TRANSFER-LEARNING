# src/predict.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import argparse

# 1. Setup Argument Parser (so we can run it from command line)
parser = argparse.ArgumentParser(description='Detect Deepfakes')
parser.add_argument('-i', '--image', type=str, required=True, help='Path to image file')
args = parser.parse_args()

# 2. Load the trained brain
# We need to tell Keras about the custom 'Meso4' layer logic if we load the full model, 
# or simpler: just load the h5 file since standard layers are used.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mesonet_best.h5')

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 3. Preprocess the image (Must match training data!)
def prepare_image(image_path):
    # Load image and resize to 256x256 (same as training)
    img = load_img(image_path, target_size=(256, 256))
    
    # Convert to array and scale pixel values to [0, 1]
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    
    # Add extra dimension because model expects a batch (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 4. Predict
try:
    processed_img = prepare_image(args.image)
    prediction = model.predict(processed_img)[0][0] # Get single float value
    
    # 5. Output Results
    print("\n" + "="*30)
    print(f"🖼️  Image: {args.image}")
    print(f"📊 Score: {prediction:.4f}")
    
    # The output is between 0 and 1.
    # Usually 0 = Fake, 1 = Real (or vice versa depending on folder order).
    # In Keras 'flow_from_directory', it sorts alphabetically:
    # 0: 'fake', 1: 'real'
    
    if prediction > 0.5:
        print("✅ RESULT: REAL Face")
        confidence = (prediction - 0.5) * 2 * 100
        print(f"🔒 Confidence: {confidence:.2f}%")
    else:
        print("⚠️ RESULT: FAKE Face")
        confidence = (0.5 - prediction) * 2 * 100
        print(f"🚨 Confidence: {confidence:.2f}%")
    print("="*30 + "\n")

except Exception as e:
    print(f"Error: {e}")