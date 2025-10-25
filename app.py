import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("shoe_model_v2.keras")
    return model

model = load_model()

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="Fake Shoe Detector", layout="centered")
st.title("👟 Fake Shoe Detection App")
st.write("Upload an image of a shoe to predict whether it is **Real** or **Fake**.")

# ------------------------------
# Image Upload
# ------------------------------
uploaded_file = st.file_uploader("Choose a shoe image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Shoe Image", use_container_width=True)
    st.write("")

    # Preprocess the image
    img = img.resize((224, 224))  # change this to match your model input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    # Interpret result (assuming binary classification: 0=fake, 1=real)
    if score > 0.5:
        st.success(f"✅ **Real Shoe** (Confidence: {score*100:.2f}%)")
    else:
        st.error(f"❌ **Fake Shoe** (Confidence: {(1-score)*100:.2f}%)")
