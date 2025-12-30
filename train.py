# src/train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mesonet import Meso4  # Importing the brain we made in file 1
import os

# --- CONFIGURATION ---
# We point to the processed_data folder located one level up from 'src'
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed_data')
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "valid")

# Settings
BATCH_SIZE = 32
EPOCHS = 50  # Start with 1 to test. Change to 10 or 30 later for real training.

# --- 1. PREPARE DATA ---
# Rescale pixel values from [0, 255] to [0, 1] for the AI
datagen = ImageDataGenerator(rescale=1.0/255.0)

print(f"Checking for data in: {TRAIN_DIR}")

# This automatically finds 'real' and 'fake' folders inside 'train'
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# This automatically finds 'real' and 'fake' folders inside 'valid'
val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 2. LOAD MODEL ---
print("Building MesoNet model...")
meso = Meso4()
model = meso.model

# --- 3. TRAIN ---
print(f"Starting Training for {EPOCHS} epoch(s)...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- 4. SAVE ---
# Create a models folder if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

save_path = os.path.join(models_dir, "mesonet_best.h5")
model.save(save_path)
print(f"✅ Success! Model saved to: {save_path}")